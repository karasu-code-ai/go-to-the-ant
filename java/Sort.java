/*
 * "Go to the Ant" — a faithful Java port of Deneubourg's ant brood sorting (§3.2).
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
 * Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997), recreating
 * Deneubourg et al. (1991) ant brood/corpse sorting.
 *
 * LENS: Java — the classical JVM agent-based-modeling lineage (MASON / Repast / NetLogo).
 * The natural idiom here is explicit objects: a Nest (the World) that owns the shared grid
 * of brood items, and a schedule that steps individually-instantiated SortAnt objects. No
 * ant runs a sorting algorithm; sorting EMERGES from four purely local rules.
 *
 * The four local rules (§3.2), preserved verbatim in spirit:
 *   1. Wander randomly around the nest (dx,dy each in {-1,0,1}, toroidal).
 *   2. Keep a SHORT memory (~10 steps) of the object types recently seen (empties included).
 *   3. Not carrying + at an object: pick it up stochastically.
 *          p(pickup) = (k+/(k+ + f))^2      -- PAPER §3.2 VERBATIM
 *      where f is the fraction of short memory holding the SAME type. (Rare type -> f small
 *      -> pick up ~surely.)
 *   4. Carrying + on empty ground: drop it stochastically.
 *          p(putdown) = (f/(k- + f))^2      -- PAPER §3.2 VERBATIM
 *      (Surrounded by the same type -> f large -> drop ~surely.)
 *   Constants (paper): k+ ~ 1, k- ~ 3 -- OPERATIONALIZED as kp=1.0, km=3.0; k- must exceed
 *   k+ or clusters dissolve faster than they form. mem ~ 10 (OPERATIONALIZED as memory=10).
 *
 * Clustering = mean fraction of the 8 toroidal neighbours that share an item's type
 * (OPERATIONALIZED quality metric: 0 = scattered, 1 = perfectly sorted).
 *
 * DEPENDENCY-FREE: standard library only (RNG is a hand-rolled SplitMix64, shared verbatim
 * with the foraging port so the sequential ports stay bit-comparable).
 */

public class Sort {

    static final char[] TYPES = {'A', 'B', 'C'};   // kinds of brood items (larvae/eggs/cocoons)

    // ---- SplitMix64 PRNG: identical across all ports so they are directly comparable. ----
    // This is a CROSS-PORT CONVENTION; it intentionally does NOT reproduce CPython's random
    // module. The goal is that all six ports agree with EACH OTHER, consuming the RNG in the
    // same order.
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
        // randrange(n) -> int in [0,n): unsigned next() % n, matching the C/Go/Rust/JS ports.
        // (Math.floorMod / signed % would diverge whenever next()'s high bit is set, since 2^64 mod n != 0.)
        int randrange(int n) {
            return (int) (Long.remainderUnsigned(next(), (long) n));
        }
        // choice(list): list[randrange(len)]
        int choice(int[] list) {
            return list[randrange(list.length)];
        }
        // shuffle(arr): Fisher-Yates -- for i from len-1 downto 1: j=randrange(i+1); swap.
        void shuffle(int[] arr) {
            for (int i = arr.length - 1; i >= 1; i--) {
                int j = randrange(i + 1);
                int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            }
        }
    }

    static final int[] STEP3 = {-1, 0, 1};   // wander offsets (choice source, rule 1)

    static final class Nest {
        final int w, h, nPerType;
        final SplitMix64 rng;
        final char[][] grid;    // 0 == empty, else a type char

        Nest(int w, int h, int nPerType, long seed) {
            this.w = w; this.h = h; this.nPerType = nPerType;
            this.rng = new SplitMix64(seed);
            this.grid = new char[h][w];   // default '\0' == empty

            // cells = [(x,y) for y in range(h) for x in range(w)], then shuffle, then scatter.
            int[] cells = new int[w * h];         // encode as y*w + x
            for (int y = 0, k = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    cells[k++] = y * w + x;
            rng.shuffle(cells);
            int i = 0;
            for (char t : TYPES) {                // scatter each type at random
                for (int c = 0; c < nPerType; c++) {
                    int enc = cells[i++];
                    grid[enc / w][enc % w] = t;
                }
            }
        }

        // Mean fraction of 8 toroidal neighbours that share an item's type.
        double clustering() {
            int tot = 0;
            double same = 0.0;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    char t = grid[y][x];
                    if (t == 0) continue;
                    int neigh = 0, simt = 0;
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            if (dx == 0 && dy == 0) continue;
                            int nx = Math.floorMod(x + dx, w);
                            int ny = Math.floorMod(y + dy, h);
                            char nt = grid[ny][nx];
                            if (nt != 0) {
                                neigh += 1;
                                if (nt == t) simt += 1;
                            }
                        }
                    }
                    if (neigh > 0) {
                        tot += 1;
                        same += (double) simt / neigh;
                    }
                }
            }
            return same / Math.max(tot, 1);
        }
    }

    static final class SortAnt {
        final Nest n;
        int x, y;
        char carry;             // 0 == not carrying
        final char[] mem;       // ring buffer of recently-seen types (0 == empty cell seen)
        int memLen, memHead;    // memLen entries, oldest at memHead; deque(maxlen=mem)
        final int memCap;
        final double kp, km;

        SortAnt(Nest nest, int mem, double kp, double km) {
            this.n = nest;
            this.x = nest.rng.randrange(nest.w);   // matches SortAnt.__init__ RNG order
            this.y = nest.rng.randrange(nest.h);
            this.carry = 0;
            this.memCap = mem;
            this.mem = new char[mem];
            this.memLen = 0; this.memHead = 0;
            this.kp = kp; this.km = km;
        }

        // rule 2: record a seen type into the bounded short memory.
        void remember(char here) {
            if (memLen < memCap) {
                mem[(memHead + memLen) % memCap] = here;
                memLen++;
            } else {
                mem[memHead] = here;
                memHead = (memHead + 1) % memCap;
            }
        }

        // f = fraction of memory holding the SAME type t.
        double f(char t) {
            if (memLen == 0) return 0.0;
            int c = 0;
            for (int i = 0; i < memLen; i++) {
                if (mem[(memHead + i) % memCap] == t) c++;
            }
            return (double) c / memLen;
        }

        void step() {
            // rule 1: wander (toroidal). choice(dx) then choice(dy) -- exact RNG order.
            x = Math.floorMod(x + n.rng.choice(STEP3), n.w);
            y = Math.floorMod(y + n.rng.choice(STEP3), n.h);
            char here = n.grid[y][x];
            remember(here);                         // rule 2 (record even empties)
            if (carry == 0) {
                if (here != 0) {                    // rule 3: maybe pick up
                    double fv = f(here);
                    double p = Math.pow(kp / (kp + fv), 2);   // PAPER §3.2 VERBATIM: p(pickup)=(k+/(k++f))^2
                    if (n.rng.random() < p) {
                        carry = here; n.grid[y][x] = 0;
                    }
                }
            } else {
                if (here == 0) {                    // rule 4: maybe drop
                    double fv = f(carry);
                    double p = Math.pow(fv / (km + fv), 2);   // PAPER §3.2 VERBATIM: p(putdown)=(f/(k-+f))^2
                    if (n.rng.random() < p) {
                        n.grid[y][x] = carry; carry = 0;
                    }
                }
            }
        }
    }

    static void render(Nest nest) {
        StringBuilder out = new StringBuilder();
        for (int y = 0; y < nest.h; y++) {
            for (int x = 0; x < nest.w; x++) {
                char c = nest.grid[y][x];
                out.append(c == 0 ? '.' : c);
            }
            out.append('\n');
        }
        System.out.print(out);
    }

    static void run(int ticks, int nAnts, long seed, boolean verbose) {
        Nest nest = new Nest(40, 24, 90, seed);
        SortAnt[] ants = new SortAnt[nAnts];
        for (int i = 0; i < nAnts; i++) ants[i] = new SortAnt(nest, 10, 1.0, 3.0);

        if (verbose) {
            System.out.println("BEFORE (random scatter):\n");
            render(nest);
            System.out.printf("%ninitial clustering: %.3f%n", nest.clustering());
        }

        StringBuilder hist = new StringBuilder();
        int sampleEvery = Math.max(1, ticks / 12);
        for (int t = 0; t < ticks; t++) {
            for (SortAnt a : ants) a.step();
            if (t % sampleEvery == 0) {
                if (hist.length() > 0) hist.append(' ');
                hist.append(String.format("%.2f", nest.clustering()));
            }
        }

        if (verbose) {
            System.out.println("\nAFTER (emergent sorting):\n");
            render(nest);
            System.out.printf("%nfinal clustering: %.3f%n", nest.clustering());
            System.out.println("clustering(t): " + hist);
        }
    }

    public static void main(String[] args) {
        int ticks = 120000, ants = 40;
        long seed = 0;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--ticks": ticks = Integer.parseInt(args[++i]); break;
                case "--ants":  ants  = Integer.parseInt(args[++i]); break;
                case "--seed":  seed  = Long.parseLong(args[++i]); break;
                default: /* ignore unknown args */ break;
            }
        }
        run(ticks, ants, seed, true);
    }
}
