/*
 * "Go to the Ant" §3.5 — Birds & Fish: Flocking (Reynolds 1987, Heppner 1990),
 * a faithful Java port of the reference recreated from Parunak's survey.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
 * Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
 *
 * LENS: Java — the classical JVM agent-based-modeling lineage (MASON, Repast, NetLogo).
 * The idiom is explicit objects under a schedule: a World that OWNS the shared state
 * arrays (px, py, vx, vy) and a swarm of individually-instantiated Boid objects, each
 * carrying only its own index and the three local steering rules. The World steps every
 * Boid to compute a next velocity, THEN commits positions — a two-phase schedule so no
 * boid sees a half-updated flock. There is no leader and no central plan.
 *
 * Reynolds' three local rules (§3.5), which are the paper's:
 *   1. SEPARATION — keep a minimum distance from the nearest birds (avoid collisions).
 *   2. ALIGNMENT  — match velocity (speed + heading) to nearby birds.
 *   3. COHESION   — steer toward the centre of the local flock.
 * A single coherent, banking flock EMERGES from these three local urges.
 *
 * PROVENANCE: the three rules are the paper's (Reynolds' "boids"). The perception radius,
 * the separation distance, and the three weights are OPERATIONALIZED — Parunak's paper
 * lists the rules but gives no numbers (Reynolds 1987 is the primary source for tuned
 * constants). NO per-step randomness: the run is fully deterministic given the random
 * init, so cross-port identity depends only on matching the init-RNG order and the
 * neighbour-sum order (iterate j in index order).
 *
 * DEPENDENCY-FREE: standard library only (RNG is a hand-rolled SplitMix64).
 */

public class Flocking {

    // ---- SplitMix64 PRNG: identical across all ports so they are directly comparable. ----
    static final class SplitMix64 {
        private long state;
        SplitMix64(long seed) { this.state = seed; }
        long next() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }
        // random_float() in [0,1): (next() >> 11) * 2^-53
        double random() {
            return (next() >>> 11) * (1.0 / 9007199254740992.0);
        }
        // uniform(a,b): a + random_float()*(b-a)
        double uniform(double a, double b) {
            return a + random() * (b - a);
        }
    }

    // The World owns the shared flock state; Boids read/write into it under the schedule.
    static final class World {
        final int n, w, h;
        final SplitMix64 rng;
        final double[] px, py, vx, vy;   // shared arrays: position and (unit-speed) velocity
        final double[] nvx, nvy;         // scratch for the two-phase (compute-then-commit) step
        final double perc, sepR, wsep, wali, wcoh, vmax, turn;

        World(int n, int w, int h, long seed,
              double perc, double sepR, double wsep, double wali, double wcoh,
              double vmax, double turn) {
            this.n = n; this.w = w; this.h = h;
            this.rng = new SplitMix64(seed);
            this.perc = perc; this.sepR = sepR;
            this.wsep = wsep; this.wali = wali; this.wcoh = wcoh;
            this.vmax = vmax; this.turn = turn;
            this.px = new double[n]; this.py = new double[n];
            this.vx = new double[n]; this.vy = new double[n];
            this.nvx = new double[n]; this.nvy = new double[n];
            // Init: px,py uniform in the box; heading uniform; vx=cos, vy=sin.
            // RNG consumption order MUST match the Python reference exactly:
            // all px, then all py, then all angles.
            for (int i = 0; i < n; i++) px[i] = rng.uniform(0, w);
            for (int i = 0; i < n; i++) py[i] = rng.uniform(0, h);
            for (int i = 0; i < n; i++) {
                double a = rng.uniform(0, 2 * Math.PI);
                vx[i] = Math.cos(a); vy[i] = Math.sin(a);
            }
        }

        // Order parameter: |mean heading| / vmax. 0 = disordered, 1 = one coherent flock.
        double polarization() {
            double mx = 0, my = 0;
            for (int i = 0; i < n; i++) { mx += vx[i]; my += vy[i]; }
            mx /= n; my /= n;
            return Math.hypot(mx, my) / vmax;
        }

        // Two-phase schedule: every Boid computes its next velocity into (nvx,nvy) from the
        // frozen current state, THEN we commit velocities and move positions toroidally.
        double step(Boid[] boids) {
            for (int i = 0; i < n; i++) { nvx[i] = vx[i]; nvy[i] = vy[i]; }
            for (Boid b : boids) b.computeNextVelocity();
            for (int i = 0; i < n; i++) {
                vx[i] = nvx[i]; vy[i] = nvy[i];
                px[i] = mod(px[i] + vx[i], w);
                py[i] = mod(py[i] + vy[i], h);
            }
            return polarization();
        }

        // Python's % on floats always returns a result with the divisor's sign (non-negative here).
        static double mod(double a, double m) {
            double r = a % m;
            return r < 0 ? r + m : r;
        }
    }

    // A single bird. Holds only its index and a back-reference to the shared World — all
    // steering is LOCAL (neighbours inside the perception radius).
    static final class Boid {
        final World world;
        final int i;
        Boid(World world, int i) { this.world = world; this.i = i; }

        void computeNextVelocity() {
            final World wd = world;
            final int n = wd.n;
            final double[] px = wd.px, py = wd.py, vx = wd.vx, vy = wd.vy;
            final double p2 = wd.perc * wd.perc, s2 = wd.sepR * wd.sepR;
            double sx = 0, sy = 0, ax = 0, ay = 0, cx = 0, cy = 0;
            int cnt = 0;
            // Iterate j in index order to keep the neighbour-sum bit-identical across ports.
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                double dx = px[j] - px[i], dy = py[j] - py[i];
                // toroidal delta (VERBATIM: dx -= w*round(dx/w)); Math.rint == Python round (half-to-even)
                dx -= wd.w * Math.rint(dx / wd.w);
                dy -= wd.h * Math.rint(dy / wd.h);
                double d2 = dx * dx + dy * dy;
                if (d2 > p2) continue;
                cnt++;
                ax += vx[j]; ay += vy[j];          // rule 2: alignment (avg neighbour velocity)
                cx += dx; cy += dy;                // rule 3: cohesion (toward neighbour centre)
                if (d2 < s2 && d2 > 1e-9) {         // rule 1: separation (push from the close ones)
                    sx -= dx / d2; sy -= dy / d2;
                }
            }
            if (cnt == 0) return;                   // no neighbours: keep current velocity
            ax /= cnt; ay /= cnt; cx /= cnt; cy /= cnt;
            // NORMALIZE each urge to a unit vector so the three weights are actually comparable
            // (otherwise the position-scale cohesion vector swamps the velocity-scale alignment one).
            double sux = 0, suy = 0, aux = 0, auy = 0, cux = 0, cuy = 0;
            double m;
            m = Math.hypot(sx, sy);              if (m > 1e-9) { sux = sx / m; suy = sy / m; }
            double avx = ax - vx[i], avy = ay - vy[i];
            m = Math.hypot(avx, avy);            if (m > 1e-9) { aux = avx / m; auy = avy / m; }
            m = Math.hypot(cx, cy);             if (m > 1e-9) { cux = cx / m; cuy = cy / m; }
            double accx = wd.wsep * sux + wd.wali * aux + wd.wcoh * cux;
            double accy = wd.wsep * suy + wd.wali * auy + wd.wcoh * cuy;
            double nvx = vx[i] + wd.turn * accx, nvy = vy[i] + wd.turn * accy;
            double sp = Math.hypot(nvx, nvy);
            if (sp == 0.0) sp = 1.0;                // cap speed to vmax
            wd.nvx[i] = nvx / sp * wd.vmax;
            wd.nvy[i] = nvy / sp * wd.vmax;
        }
    }

    static void render(World wd) {
        char[] arrow = {'→', '↗', '↑', '↖', '←', '↙', '↓', '↘'};
        char[][] grid = new char[wd.h][wd.w];
        for (int y = 0; y < wd.h; y++) for (int x = 0; x < wd.w; x++) grid[y][x] = ' ';
        for (int i = 0; i < wd.n; i++) {
            int x = ((int) wd.px[i] % wd.w + wd.w) % wd.w;
            int y = ((int) wd.py[i] % wd.h + wd.h) % wd.h;
            double a = Math.atan2(wd.vy[i], wd.vx[i]);
            int k = ((int) Math.rint(a / (Math.PI / 4)) % 8 + 8) % 8;
            grid[y][x] = arrow[k];
        }
        System.out.println();
        System.out.println("Flock (each bird points along its heading — watch them align):");
        System.out.println();
        StringBuilder out = new StringBuilder();
        for (int y = 0; y < wd.h; y++) {
            out.append(new String(grid[y])).append('\n');
        }
        System.out.print(out);
    }

    static void run(int ticks, int n, long seed) {
        // n=90; world 90x48; perc=8; sep_r=3; wsep=1.3; wali=1.5; wcoh=0.85; vmax=1.0; turn=0.35.
        World wd = new World(n, 90, 48, seed, 8.0, 3.0, 1.3, 1.5, 0.85, 1.0, 0.35);
        Boid[] boids = new Boid[n];
        for (int i = 0; i < n; i++) boids[i] = new Boid(wd, i);

        StringBuilder hist = new StringBuilder();
        hist.append(String.format("%.2f", wd.polarization()));
        int sample = Math.max(1, ticks / 12);
        for (int t = 0; t < ticks; t++) {
            double p = wd.step(boids);
            if (t % sample == 0) {
                hist.append(' ').append(String.format("%.2f", p));
            }
        }
        render(wd);
        System.out.println();
        System.out.printf("polarization (flock alignment): %.3f  (0 = chaos, 1 = one flock)%n",
                wd.polarization());
        System.out.println("polarization(t): " + hist);
    }

    public static void main(String[] args) {
        int ticks = 600, birds = 90;
        long seed = 0;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--ticks": ticks = Integer.parseInt(args[++i]); break;
                case "--birds": birds = Integer.parseInt(args[++i]); break;
                case "--ants":  birds = Integer.parseInt(args[++i]); break;  // alias for parity
                case "--seed":  seed  = Long.parseLong(args[++i]); break;
                default: /* ignore unknown args */ break;
            }
        }
        run(ticks, birds, seed);
    }
}
