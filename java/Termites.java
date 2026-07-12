/*
 * "Go to the Ant" — a faithful Java port of Parunak's termite nest-building swarm (§3.3).
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
 * Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997). The termite
 * construction rules trace back to Kugler, Turvey et al. (1990), §3.3.
 *
 * LENS: Java — the classical JVM agent-based-modeling lineage (MASON / Repast / NetLogo).
 * The natural idiom is explicit objects: a Mound (the World) that owns the two shared
 * scalar fields, and a schedule that steps a swarm of individually-instantiated Termite
 * objects. This is the object-per-agent, schedule-driven view of stigmergy.
 *
 * The three local termite rules (§3.3), preserved in spirit:
 *   1. Metabolize bodily waste, which carries pheromone. The waste IS the building material.
 *   2. Wander randomly, but prefer the direction of the strongest local pheromone.
 *   3. Each step, decide STOCHASTICALLY whether to deposit the current load. p(deposit)
 *      rises with LOCAL pheromone density AND the amount carried. A full termite drops
 *      even with no nearby deposit; a termite in a high local concentration drops even a
 *      small load.
 * Because pheromone DECAYS, the freshest deposits (the centre of a growing pile) smell
 * strongest, so piles climb upward into COLUMNS rather than spreading. No termite plans
 * the mound.
 *
 * TWO fields (the whole point — communication THROUGH the environment):
 *   mass:  persistent structural mass (what you see; never decays).
 *   scent: decaying pheromone that biases wandering and the deposit decision.
 *
 * DEPENDENCY-FREE: standard library only (RNG is a hand-rolled SplitMix64, identical
 * across all ports so their metrics are directly comparable).
 */

public class Termites {

    // The 8 neighbours in THIS exact order (must match the reference for cross-port identity).
    static final int[][] DIRS = {
        {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}
    };

    // ---- SplitMix64 PRNG: identical across all ports so they are directly comparable. ----
    // CROSS-PORT CONVENTION: this intentionally does NOT reproduce CPython's random module;
    // the goal is that the sequential ports agree with EACH OTHER, not with Python.
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
        // randrange(n) -> int in [0,n): UNSIGNED next() % n, matching the C/Go/Rust/JS ports.
        // (signed % with a +n guard diverges whenever next()'s high bit is set, since 2^64 mod n != 0.)
        int randrange(int n) {
            return (int) Long.remainderUnsigned(next(), n);
        }
    }

    static final class Mound {
        final int w, h;
        final SplitMix64 rng;
        final double[][] mass;    // persistent structure (viz); never decays
        final double[][] scent;   // decaying pheromone (biases the walk + deposit)
        final double decay;

        Mound(int w, int h, long seed, double decay) {
            this.w = w; this.h = h;
            this.rng = new SplitMix64(seed);
            this.mass = new double[h][w];
            this.scent = new double[h][w];
            this.decay = decay;
        }

        // The scent field dissipates every tick (the entropy leak that makes piles CLIMB
        // rather than spread — fresh deposits at a pile's core stay the strongest smell).
        void evaporate() {
            double keep = 1.0 - decay;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    scent[y][x] *= keep;
                }
            }
        }

        // Count distinct COLUMNS = toroidal local maxima with mass above 0.15*peak.
        // Returns {count, peak}.
        double[] columns() {
            double peak = 0.0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if (mass[y][x] > peak) peak = mass[y][x];
            if (peak <= 0) return new double[]{0, 0.0};
            double cut = peak * 0.15;
            int cnt = 0;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    double v = mass[y][x];
                    if (v < cut) continue;
                    boolean localMax = true;
                    for (int[] d : DIRS) {
                        int nx = ((x + d[0]) % w + w) % w;
                        int ny = ((y + d[1]) % h + h) % h;
                        if (v < mass[ny][nx]) { localMax = false; break; }
                    }
                    if (localMax) cnt++;
                }
            }
            return new double[]{cnt, peak};
        }
    }

    static final class Termite {
        final Mound m;
        int x, y;
        double load;
        final double metab, maxload;

        Termite(Mound m, double metab, double maxload) {
            this.m = m;
            this.x = m.rng.randrange(m.w);   // RNG order: x then y (matches reference)
            this.y = m.rng.randrange(m.h);
            this.load = 0.0;
            this.metab = metab; this.maxload = maxload;
        }

        void step() {
            // rule 1: metabolize -> waste (building material) accumulates, capped at maxload.
            load = Math.min(maxload, load + metab);

            // rule 2: wander over the 8 toroidal neighbours, biased toward strongest scent.
            double[] wts = new double[8];
            int[] nxs = new int[8];
            int[] nys = new int[8];
            double tot = 0.0;
            for (int i = 0; i < 8; i++) {
                int nx = ((x + DIRS[i][0]) % m.w + m.w) % m.w;
                int ny = ((y + DIRS[i][1]) % m.h + m.h) % m.h;
                double wgt = 1.0 + m.scent[ny][nx] * 3.0;
                wts[i] = wgt; nxs[i] = nx; nys[i] = ny;
                tot += wgt;
            }
            double r = m.rng.random() * tot;
            for (int i = 0; i < 8; i++) {
                r -= wts[i];
                if (r <= 0) { x = nxs[i]; y = nys[i]; break; }
            }

            // rule 3: stochastic deposit — rises with LOCAL scent AND load; a full termite
            // always drops. OPERATIONALIZED: paper §3.3 gives NO formula, only "prob rises
            // with local density AND load"; the linear blend below is our operationalization.
            double local = m.scent[y][x];
            double p = Math.min(1.0, 0.01 + 0.55 * (load / maxload) + 0.20 * local);
            if (load >= maxload || m.rng.random() < p) {
                m.mass[y][x] += load;
                m.scent[y][x] += load;
                load = 0.0;
            }
        }
    }

    static void render(Mound m) {
        double peak = 0.0;
        for (int y = 0; y < m.h; y++)
            for (int x = 0; x < m.w; x++)
                if (m.mass[y][x] > peak) peak = m.mass[y][x];
        if (peak == 0.0) peak = 1.0;

        final String shades = " .:-=+*#%@";
        System.out.println();
        System.out.println("Termite mound (top-down mass density — columns emerge as bright cores):");
        System.out.println();
        StringBuilder out = new StringBuilder();
        for (int y = 0; y < m.h; y++) {
            for (int x = 0; x < m.w; x++) {
                int lvl = (int) (m.mass[y][x] / peak * (shades.length() - 1));
                if (lvl < 0) lvl = 0;
                if (lvl > shades.length() - 1) lvl = shades.length() - 1;
                out.append(shades.charAt(lvl));
            }
            out.append('\n');
        }
        System.out.print(out);
    }

    static void run(int ticks, int n, long seed, double decay) {
        Mound mound = new Mound(58, 34, seed, decay);
        Termite[] termites = new Termite[n];
        for (int i = 0; i < n; i++) termites[i] = new Termite(mound, 0.4, 6.0);

        StringBuilder hist = new StringBuilder();
        int step = Math.max(1, ticks / 12);
        for (int t = 0; t < ticks; t++) {
            for (Termite tm : termites) tm.step();
            mound.evaporate();
            if (t % step == 0) {
                if (hist.length() > 0) hist.append(' ');
                hist.append((int) mound.columns()[0]);
            }
        }

        render(mound);
        double[] cp = mound.columns();
        System.out.println();
        System.out.printf("distinct columns (local maxima): %d | tallest column mass: %.0f%n",
                (int) cp[0], cp[1]);
        System.out.println("columns(t): " + hist);
    }

    public static void main(String[] args) {
        int ticks = 40000, termites = 70;
        double decay = 0.02;
        long seed = 0;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--ticks":    ticks    = Integer.parseInt(args[++i]); break;
                case "--termites": termites = Integer.parseInt(args[++i]); break;
                case "--ants":     termites = Integer.parseInt(args[++i]); break; // alias
                case "--decay":    decay    = Double.parseDouble(args[++i]); break;
                case "--seed":     seed     = Long.parseLong(args[++i]); break;
                default: /* ignore unknown args */ break;
            }
        }
        run(ticks, termites, seed, decay);
    }
}
