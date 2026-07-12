/*
 * "Go to the Ant" — a faithful Java port of Wasp Task Differentiation (§3.4).
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
 * Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997), recreating
 * Theraulaz, Goss, Gervet & Deneubourg's Polistes wasp model (1991).
 *
 * LENS: Java — the classical JVM agent-based-modeling lineage (MASON, Repast, NetLogo).
 * The natural idiom is explicit objects: a Colony (the World) owns the shared population
 * state and the brood demand D, and a schedule steps a swarm of Wasp instances that each
 * carry two scalars — Force (mobility) and Threshold (brood sensitivity). No wasp computes
 * the caste proportions; three castes SELF-SEPARATE in the (Force, Threshold) plane.
 *
 * The three interacting rules (Parunak §3.4):
 *   1. FACE-OFFS.  When two wasps meet, j beats i with the Fermi probability
 *        p = 1/(1 + e^(h*(F_i - F_j)))                          [PAPER VERBATIM]
 *      The higher force usually wins; a quantum of Force passes loser -> winner (conserved).
 *   2. BROOD DEMAND.  D(t) = D(t-1) + appetite - W, W = count of mobile foragers who worked.
 *   3. FORAGE?  A wasp near the brood forages with Fermi probability
 *        p = 1/(1 + e^(hf*(sig_j - D)))                          [PAPER VERBATIM]
 *      Foraging LOWERS its threshold sig by xi (learning); not foraging RAISES it by phi.
 *
 * OPERATIONALIZED (this is the most provenance-heavy sim; preserved from the Python source):
 *   - ENTROPY LEAK (§4.6): force DISSIPATES and REGENERATES among the wasps. A steady
 *     leak+gen bounds the hierarchy naturally (equilibrium mean ~ gen/leak), so no ad-hoc
 *     force CAP is needed to stop one super-wasp. Tradeoff: dissipating force shifts the
 *     force distribution and couples into the threshold balance, so the Chief's high-threshold
 *     detail regresses vs a hard-capped version. The two mechanisms are coupled by design.
 *   - DOMINANCE = (F/Fmax)^4 as a SPATIALITY PROXY: the paper's Chief "wanders and faces off",
 *     so it is not near the brood and is rarely stimulated -> its threshold drifts HIGH. We
 *     approximate "away dominating" by suppressing the top-force wasp's foraging; the exponent
 *     4 makes dominance ~1 only for F~Fmax (the Chief), leaving foragers (F~0.7*Fmax) untouched.
 *
 * DEPENDENCY-FREE: standard library only (RNG is a hand-rolled SplitMix64, identical across
 * all ports so they are directly comparable).
 */

public class Wasps {

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
        // randrange(n) -> int in [0, n)
        int randrange(int n) {
            return (int) (Long.remainderUnsigned(next(), n));
        }
        // uniform(a,b): a + random_float()*(b-a)
        double uniform(double a, double b) {
            return a + random() * (b - a);
        }
    }

    // A single genetically-identical wasp: just its two evolving scalars. The interesting
    // behaviour lives in the Colony's per-tick schedule (the ABM "schedule" idiom).
    static final class Wasp {
        double force;      // F: mobility — a low-force wasp is stimulated but cannot travel to hunt
        double threshold;  // sig: brood-response threshold (low = attentive)
        Wasp(double force, double threshold) { this.force = force; this.threshold = threshold; }
    }

    static final class Colony {
        static final double SIGMAX = 4.0;

        final SplitMix64 rng;
        final int n;
        final Wasp[] wasps;
        final double h, hf, q;       // Fermi steepness (face-off), Fermi steepness (forage), force quantum
        final double appetite;       // brood appetite per tick
        final double xi, phi, mob;   // learning drop, forgetting rise, mobility cutoff
        final double leak, gen;      // entropy leak: force dissipation + regeneration
        double D;                    // brood demand

        Colony(int n, long seed, double h, double hf, double quantum, double appetite,
               double xi, double phi, double mob, double leak, double gen) {
            this.rng = new SplitMix64(seed);
            this.n = n;
            this.h = h; this.hf = hf; this.q = quantum;
            this.appetite = appetite;
            this.xi = xi; this.phi = phi; this.mob = mob;
            this.leak = leak; this.gen = gen;
            this.D = 2.0;
            this.wasps = new Wasp[n];
            // genetically identical: tiny initial spread only. RNG order must match Python —
            // ALL force inits first (n uniforms), THEN all threshold inits (n uniforms).
            double[] f0 = new double[n];
            for (int i = 0; i < n; i++) f0[i] = 1.0 + rng.uniform(-0.05, 0.05);
            double[] s0 = new double[n];
            for (int i = 0; i < n; i++) s0[i] = 1.6 + rng.uniform(-0.05, 0.05);
            for (int i = 0; i < n; i++) wasps[i] = new Wasp(f0[i], s0[i]);
        }

        void step() {
            final Wasp[] w = wasps;
            // rule 1: face-offs — gentle, capped, so a graded hierarchy forms (not one super-wasp)
            for (int r = 0; r < n / 3; r++) {
                int i = rng.randrange(n);
                int j = rng.randrange(n);
                if (i == j) continue;
                // PAPER §3.4 VERBATIM: p = 1/(1 + e^(h*(Fi - Fj)))
                double pj = 1.0 / (1.0 + Math.exp(h * (w[i].force - w[j].force)));
                int win, los;
                if (rng.random() < pj) { win = j; los = i; } else { win = i; los = j; }
                double t = Math.min(q, w[los].force);
                w[win].force += t; w[los].force -= t;   // force is conserved in the face-off (paper)
            }
            // ENTROPY LEAK (§4.6, PRINCIPLE APPLIED): force DISSIPATES and REGENERATES. A steady
            // leak+gen bounds the hierarchy naturally (equilibrium mean ~ gen/leak), no ad-hoc cap.
            for (int k = 0; k < n; k++) {
                w[k].force = Math.max(0.0, w[k].force * (1.0 - leak) + gen);
            }
            // rules 2 & 3: brood stimulation + foraging. Work = COUNT of mobile foragers (each brings 1).
            // SPATIALITY PROXY (operationalized): suppress the top-force wasp's foraging so its
            // threshold drifts HIGH (the Chief wanders and faces off rather than tending the brood).
            int W = 0;
            double Fmax = 0.0;
            for (int k = 0; k < n; k++) if (w[k].force > Fmax) Fmax = w[k].force;
            if (Fmax == 0.0) Fmax = 1.0;
            for (int k = 0; k < n; k++) {
                // PAPER §3.4 VERBATIM: p = 1/(1 + e^(hf*(sig - D)))
                double pf = 1.0 / (1.0 + Math.exp(hf * (w[k].threshold - D)));
                double ratio = w[k].force / Fmax;
                double dom = ratio * ratio * ratio * ratio;   // (F/Fmax)^4 ~ 1 only for the single top wasp
                if (rng.random() < pf * (1.0 - dom)) {        // stimulated AND not away dominating
                    w[k].threshold = Math.max(0.0, w[k].threshold - xi);   // learns: threshold drops
                    if (w[k].force > mob) W++;                             // mobile enough to actually hunt
                } else {
                    w[k].threshold = Math.min(SIGMAX, w[k].threshold + phi); // forgets: threshold rises
                }
            }
            D = Math.max(0.0, D + appetite - W);
        }

        int chief() {
            int c = 0;
            for (int k = 1; k < n; k++) if (wasps[k].force > wasps[c].force) c = k;
            return c;
        }

        // median threshold, matching Python's sorted(sig)[n//2]
        double medianThreshold() {
            double[] s = new double[n];
            for (int k = 0; k < n; k++) s[k] = wasps[k].threshold;
            java.util.Arrays.sort(s);
            return s[n / 2];
        }

        // Returns caste index per wasp: 0 = Chief, 1 = Forager, 2 = Nurse.
        int[] castes() {
            int chief = chief();
            double smed = medianThreshold();
            int[] out = new int[n];
            for (int k = 0; k < n; k++) {
                if (k == chief) out[k] = 0;
                else if (wasps[k].force > mob && wasps[k].threshold <= smed) out[k] = 1; // mobile + responsive
                else out[k] = 2;                                                          // stays with the brood
            }
            return out;
        }
    }

    static void run(int ticks, int n, long seed) {
        Colony c = new Colony(n, seed, 1.1, 3.0, 0.10, 0.075 * n,
                              0.02, 0.012, 1.6, 0.004, 0.005);
        StringBuilder hist = new StringBuilder();
        int sample = Math.max(1, ticks / 12);
        for (int t = 0; t < ticks; t++) {
            c.step();
            if (t % sample == 0) {
                int[] g = c.castes();
                int f = 0, ns = 0;
                for (int k = 0; k < n; k++) { if (g[k] == 1) f++; else if (g[k] == 2) ns++; }
                if (hist.length() > 0) hist.append(' ');
                hist.append(f).append('/').append(ns);
            }
        }

        int[] g = c.castes();
        int chief = c.chief();
        String[] names = {"Chief", "Forager", "Nurse"};
        System.out.println("Emergent castes from " + n + " genetically identical wasps (" + ticks + " ticks):");
        System.out.println();
        for (int caste = 0; caste < 3; caste++) {
            int cnt = 0;
            double sumF = 0.0, sumS = 0.0;
            for (int k = 0; k < n; k++) {
                if (g[k] == caste) { cnt++; sumF += c.wasps[k].force; sumS += c.wasps[k].threshold; }
            }
            if (cnt == 0) continue;
            System.out.printf("  %-8s n=%3d   mean Force %5.2f   mean Threshold %5.2f%n",
                              names[caste], cnt, sumF / cnt, sumS / cnt);
        }
        double popMean = 0.0;
        for (int k = 0; k < n; k++) popMean += c.wasps[k].force;
        popMean /= n;
        System.out.println();
        System.out.printf("  Chief force %.2f (pop mean %.2f), threshold %.2f%n",
                          c.wasps[chief].force, popMean, c.wasps[chief].threshold);
        System.out.println("  Forager/Nurse split(t): " + hist);
        landscape(c);
    }

    // ASCII scatter of the population in (Force -> x, Threshold -> y) space.
    static void landscape(Colony c) {
        final int cols = 48, rows = 16;
        int n = c.n;
        double fmn = Double.MAX_VALUE, fmx = -Double.MAX_VALUE;
        double smn = Double.MAX_VALUE, smx = -Double.MAX_VALUE;
        for (int k = 0; k < n; k++) {
            double f = c.wasps[k].force, s = c.wasps[k].threshold;
            if (f < fmn) fmn = f; if (f > fmx) fmx = f;
            if (s < smn) smn = s; if (s > smx) smx = s;
        }
        char[][] grid = new char[rows][cols];
        for (char[] row : grid) java.util.Arrays.fill(row, ' ');
        int chief = c.chief();
        double[] fs = new double[n];
        for (int k = 0; k < n; k++) fs[k] = c.wasps[k].force;
        java.util.Arrays.sort(fs);
        double fmed = fs[n / 2];
        for (int k = 0; k < n; k++) {
            double f = c.wasps[k].force, s = c.wasps[k].threshold;
            int x = (int) ((f - fmn) / (fmx - fmn + 1e-9) * (cols - 1));
            int y = (int) ((s - smn) / (smx - smn + 1e-9) * (rows - 1));
            char mark = (k == chief) ? 'C' : (f >= fmed ? 'F' : 'n');
            grid[rows - 1 - y][x] = mark;
        }
        System.out.println();
        System.out.println("  (F,sig) landscape — x = Force ->, y = Threshold ^ | C chief, F forager, n nurse:");
        System.out.println();
        for (char[] row : grid) System.out.println("   " + new String(row));
    }

    public static void main(String[] args) {
        int ticks = 4000, wasps = 80;
        long seed = 0;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--ticks": ticks = Integer.parseInt(args[++i]); break;
                case "--wasps": case "--ants": wasps = Integer.parseInt(args[++i]); break;
                case "--seed":  seed  = Long.parseLong(args[++i]); break;
                default: /* ignore unknown args */ break;
            }
        }
        run(ticks, wasps, seed);
    }
}
