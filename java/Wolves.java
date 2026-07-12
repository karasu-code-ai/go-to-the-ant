/*
 * "Go to the Ant" §3.6 — Wolves: Surrounding Prey (Korf 1992), a faithful Java
 * port recreated from the paper (as surveyed in Parunak 1997).
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
 * §3.6, tracing to R. Korf, "A Simple Solution to Pursuit Games" (1992).
 *
 * LENS: Java — the classical JVM agent-based-modeling lineage (MASON, Repast,
 * NetLogo). The natural idiom is explicit objects: a World (the Hunt) owns the
 * shared state (moose position + the wolf-position array) and an explicit
 * per-tick schedule steps individually-instantiated agent objects (one Moose,
 * six Wolf instances), each carrying a single local rule.
 *
 * One wolf can't kill a moose; the pack must SURROUND it, with no radios and no
 * negotiated strategy. Parunak gives two local rules (§3.6):
 *   1. MOOSE: move to the neighbouring cell FARTHEST from the nearest wolf.
 *   2. WOLVES: move to minimise  S = d(moose) - k*d(nearest other wolf)  --
 *      get CLOSE to the moose while staying FAR from each other.
 * With attraction (to prey) and repulsion (between wolves) balanced, the pack
 * encircles the moose, no communication required.
 *
 * PROVENANCE:
 *   VERBATIM (paper's): the two local rules and the score S = d(moose) - k*d(wolf).
 *   OPERATIONALIZED: the paper states six wolves capture on a HEX GRID; here a
 *   CONTINUOUS PLANE with a 24-candidate-direction search (plus staying put),
 *   speeds vm=0.6 / vw=1.0, and k=1.12. Reason: to render the pursuit on a
 *   continuous board with an ASCII view rather than the paper's discrete hexes.
 *
 * DEPENDENCY-FREE: standard library only (RNG is a hand-rolled SplitMix64,
 * identical across all ports so they are directly comparable).
 */

public class Wolves {

    static final double TAU = 2 * Math.PI;

    // ---- SplitMix64 PRNG: identical across all ports so they are directly comparable. ----
    // (Cross-port convention; intentionally NOT CPython's random module -- the goal is that
    //  the sequential ports agree with EACH OTHER, not with Python.)
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

    // The World / schedule owner. Holds the shared arrays the agents read and write.
    static final class Hunt {
        final int w, h;
        final double vm, vw, k;
        double mx, my;          // moose position
        final double[] wx, wy;  // wolf positions (the shared array)
        final int n;

        Hunt(int nWolves, int w, int h, long seed, double vm, double vw, double k) {
            this.w = w; this.h = h; this.vm = vm; this.vw = vw; this.k = k;
            this.n = nWolves;
            SplitMix64 rng = new SplitMix64(seed);
            this.mx = w / 2.0; this.my = h / 2.0;
            this.wx = new double[nWolves];
            this.wy = new double[nWolves];
            // Consume the RNG in EXACTLY the Python order: for each wolf, x then y.
            for (int i = 0; i < nWolves; i++) {
                wx[i] = rng.uniform(0, w);
                wy[i] = rng.uniform(0, h);
            }
        }

        // The 24 direction candidates plus staying put, in Python index order.
        // Returns a flat array [x0,y0, x1,y1, ...] of 25 candidate points.
        double[] cands(double x, double y, double sp) {
            double[] c = new double[2 * 25];
            for (int i = 0; i < 24; i++) {
                double a = i * TAU / 24;
                c[2 * i]     = x + sp * Math.cos(a);
                c[2 * i + 1] = y + sp * Math.sin(a);
            }
            c[48] = x; c[49] = y;
            return c;
        }

        boolean inBounds(double cx, double cy) {
            return 0 <= cx && cx < w && 0 <= cy && cy < h;
        }

        static double hypot(double dx, double dy) { return Math.sqrt(dx * dx + dy * dy); }

        // One scheduled tick: Moose rule runs first (reading the current wolf array),
        // then all wolves choose SIMULTANEOUSLY (each reads the updated moose and the
        // OLD other-wolf positions) and are committed together.
        double step() {
            // rule 1 (VERBATIM): moose flees to the in-bounds candidate whose nearest
            // wolf is FARTHEST away.
            double bx = mx, by = my, bd = -1;
            double[] mc = cands(mx, my, vm);
            for (int c = 0; c < 25; c++) {
                double cx = mc[2 * c], cy = mc[2 * c + 1];
                if (!inBounds(cx, cy)) continue;
                double d = Double.MAX_VALUE;
                for (int j = 0; j < n; j++) {
                    double dd = hypot(cx - wx[j], cy - wy[j]);
                    if (dd < d) d = dd;
                }
                if (d > bd) { bd = d; bx = cx; by = cy; }
            }
            mx = bx; my = by;

            // rule 2 (VERBATIM score S = d(moose) - k*d(wolf)): each wolf minimises S.
            // OPERATIONALIZED update order: wolves choose against the OLD other-wolf
            // positions (a simultaneous update), matching the Python reference which
            // builds a fresh list from the unmodified `W`. (Reported deviation note.)
            double[] nx = new double[n];
            double[] ny = new double[n];
            for (int i = 0; i < n; i++) {
                double wbx = wx[i], wby = wy[i], bs = 1e9;
                double[] wc = cands(wx[i], wy[i], vw);
                for (int c = 0; c < 25; c++) {
                    double cx = wc[2 * c], cy = wc[2 * c + 1];
                    if (!inBounds(cx, cy)) continue;
                    double dm = hypot(cx - mx, cy - my);
                    double dobest = Double.MAX_VALUE;
                    boolean any = false;
                    for (int j = 0; j < n; j++) {
                        if (j == i) continue;
                        double dd = hypot(cx - wx[j], cy - wy[j]);
                        if (dd < dobest) dobest = dd;
                        any = true;
                    }
                    double doo = any ? dobest : 0.0;
                    double s = dm - k * doo;
                    if (s < bs) { bs = s; wbx = cx; wby = cy; }
                }
                nx[i] = wbx; ny[i] = wby;
            }
            System.arraycopy(nx, 0, wx, 0, n);
            System.arraycopy(ny, 0, wy, 0, n);
            return gap();
        }

        // Largest angular gap (deg) between adjacent wolves as seen from the moose.
        // 360/N when evenly ringed -> surrounded; near 360 when all on one side.
        double gap() {
            if (n < 2) return 360.0;
            double[] angs = new double[n];
            for (int i = 0; i < n; i++) angs[i] = Math.atan2(wy[i] - my, wx[i] - mx);
            java.util.Arrays.sort(angs);
            double best = -1;
            for (int i = 0; i < n; i++) {
                double g = angs[(i + 1) % n] - angs[i];
                g = ((g % TAU) + TAU) % TAU;   // Python's % is non-negative
                if (g > best) best = g;
            }
            return best * 180 / Math.PI;
        }

        double nearestWolf() {
            double md = Double.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                double d = hypot(mx - wx[i], my - wy[i]);
                if (d < md) md = d;
            }
            return md;
        }
    }

    static void render(Hunt hunt) {
        char[][] grid = new char[hunt.h][hunt.w];
        for (char[] row : grid) java.util.Arrays.fill(row, ' ');
        for (int i = 0; i < hunt.n; i++) {
            int x = ((int) hunt.wx[i]) % hunt.w;
            int y = ((int) hunt.wy[i]) % hunt.h;
            if (x < 0) x += hunt.w; if (y < 0) y += hunt.h;
            grid[y][x] = 'W';
        }
        int mxi = ((int) hunt.mx) % hunt.w, myi = ((int) hunt.my) % hunt.h;
        if (mxi < 0) mxi += hunt.w; if (myi < 0) myi += hunt.h;
        grid[myi][mxi] = 'M';
        System.out.println();
        System.out.println("The hunt (M moose, W wolves -- watch the ring close):");
        System.out.println();
        StringBuilder out = new StringBuilder();
        for (char[] row : grid) out.append(new String(row)).append('\n');
        System.out.print(out);
    }

    static void run(int ticks, int nWolves, long seed) {
        Hunt hunt = new Hunt(nWolves, 80, 44, seed, 0.6, 1.0, 1.12);
        StringBuilder hist = new StringBuilder();
        hist.append(String.format("%.0f", hunt.gap()));
        int step = Math.max(1, ticks / 12);
        for (int t = 0; t < ticks; t++) {
            double g = hunt.step();
            if (t % step == 0) hist.append(' ').append(String.format("%.0f", g));
        }
        render(hunt);
        System.out.println();
        System.out.printf("largest escape gap around the moose: %.0f°  "
                + "(evenly surrounded ≈ %d°) | nearest wolf %.1f%n",
                hunt.gap(), 360 / nWolves, hunt.nearestWolf());
        System.out.println("gap°(t): " + hist);
    }

    public static void main(String[] args) {
        int ticks = 260, wolves = 6;
        long seed = 0;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--ticks":  ticks  = Integer.parseInt(args[++i]); break;
                case "--wolves": wolves = Integer.parseInt(args[++i]); break;
                case "--ants":   wolves = Integer.parseInt(args[++i]); break; // alias
                case "--seed":   seed   = Long.parseLong(args[++i]); break;
                default: /* ignore unknown args */ break;
            }
        }
        run(ticks, wolves, seed);
    }
}
