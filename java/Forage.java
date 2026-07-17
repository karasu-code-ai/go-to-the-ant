/*
 * "Go to the Ant" — a faithful Java port of Parunak's OG foraging swarm (§3.1).
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
 * Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
 *
 * LENS: Java — the classical agent-based-modeling lineage. MASON, Repast, and NetLogo
 * were all built on the JVM, so the natural idiom here is explicit objects: a World that
 * owns the two scalar pheromone fields, and a swarm of Ant instances that each carry a
 * few local rules. This is the schedule-driven, object-per-agent view of stigmergy.
 *
 * The five local ant rules (§3.1 "Ants: Path planning"), preserved verbatim in spirit:
 *   1. Avoid obstacles.
 *   2. Wander randomly, biased toward nearby pheromone (Brownian floor + weighted scent).
 *   3. If holding food, drop pheromone at a CONSTANT RATE while walking.
 *   4. If at food and not holding any, pick it up.
 *   5. If at the nest and carrying food, drop it.
 * Field law: pheromone EVAPORATES every tick (the entropy leak), so dead trails fade.
 * No ant plans a route; the network EMERGES from deposit + evaporation + weighted walk.
 *
 * TWO local pheromone fields (the whole point — communication THROUGH the environment):
 *   food_pher: laid by CARRIERS, followed by SEARCHERS.
 *   home_pher: emitted+diffused by the NEST, followed by CARRIERS. No ant knows where the
 *              nest is — carriers just climb the local home gradient.
 *
 * DEPENDENCY-FREE: standard library only (RNG is a hand-rolled SplitMix64).
 */

public class Forage {

    // The 8 neighbours in THIS exact order (must match the reference).
    static final int[][] DIRS = {
        {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}
    };

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
    }

    static final class World {
        final int w, h;
        final SplitMix64 rng;
        final double[][] foodPher;   // laid by carriers; followed by searchers
        final double[][] homePher;   // laid/diffused by the nest; followed by carriers
        final boolean[][] obstacle;
        final int nestX, nestY;
        final int foodX, foodY;
        final int region;
        long foodQty;                // effectively unlimited source
        int deliveries;

        World(int w, int h, long seed, boolean useWall, int region) {
            this.w = w; this.h = h;
            this.rng = new SplitMix64(seed);
            this.foodPher = new double[h][w];
            this.homePher = new double[h][w];
            this.obstacle = new boolean[h][w];
            this.nestX = 5; this.nestY = h / 2;
            this.foodX = w - 6; this.foodY = h / 2;
            this.region = region;
            this.foodQty = 1_000_000_000L;
            this.deliveries = 0;
            if (useWall) {                      // wall with a single gap -> the swarm must ROUTE
                int wallx = w / 2;
                int gap = 4 + (int) (rng.random() * (h - 8)); // approximate randint(4, h-5)
                for (int y = 0; y < h; y++) {
                    if (Math.abs(y - gap) > 2) obstacle[y][wallx] = true;
                }
            }
        }

        boolean free(int x, int y) {
            return x >= 0 && x < w && y >= 0 && y < h && !obstacle[y][x];
        }

        boolean atFood(int x, int y) {
            return Math.abs(x - foodX) <= region && Math.abs(y - foodY) <= region;
        }

        boolean atNest(int x, int y) {
            return Math.abs(x - nestX) <= region && Math.abs(y - nestY) <= region;
        }

        // Both fields dissipate every tick (the entropy leak that erases dead trails).
        void evaporate(double rate) {
            double keep = 1.0 - rate;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    foodPher[y][x] *= keep;
                    homePher[y][x] *= keep;
                }
            }
        }

        // The NEST is a home-pheromone SOURCE; the marker DIFFUSES outward (Brownian, §4.6)
        // into a gradient that points home from everywhere. Carriers read only the LOCAL
        // gradient — no global nest-direction anywhere in the model.
        void emitAndDiffuseHome() {
            for (int dy = -region; dy <= region; dy++) {
                for (int dx = -region; dx <= region; dx++) {
                    int x = nestX + dx, y = nestY + dy;
                    if (free(x, y)) homePher[y][x] += 6.0;
                }
            }
            // Jacobi (simultaneous) diffusion pass over a copy, matching the reference.
            double[][] nxt = new double[h][w];
            for (int y = 0; y < h; y++) {
                System.arraycopy(homePher[y], 0, nxt[y], 0, w);
            }
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    if (!free(x, y)) continue;
                    double s = homePher[y][x];
                    int c = 1;
                    for (int[] d : DIRS) {
                        int xx = x + d[0], yy = y + d[1];
                        if (free(xx, yy)) { s += homePher[yy][xx]; c += 1; }
                    }
                    nxt[y][x] = s / c;
                }
            }
            for (int y = 0; y < h; y++) {
                System.arraycopy(nxt[y], 0, homePher[y], 0, w);
            }
        }

        // The food trail SPREADS a little (Brownian breadth, §3.1/§4.6): nearby sub-trails
        // "merge together into a trace." Local stencil over FREE neighbours only (obstacle-aware,
        // non-toroidal); draws no rng. new = old + D*(mean_free_nbrs - old). Jacobi over a copy.
        void diffuseFood(double D) {
            if (D <= 0.0) return;
            double[][] nxt = new double[h][w];
            for (int y = 0; y < h; y++) {
                System.arraycopy(foodPher[y], 0, nxt[y], 0, w);
            }
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    if (!free(x, y)) continue;
                    double s = 0.0;
                    int c = 0;
                    for (int[] d : DIRS) {
                        int xx = x + d[0], yy = y + d[1];
                        if (free(xx, yy)) { s += foodPher[yy][xx]; c += 1; }
                    }
                    if (c > 0) {
                        double cur = foodPher[y][x];
                        nxt[y][x] = cur + D * (s / c - cur);
                    }
                }
            }
            for (int y = 0; y < h; y++) {
                System.arraycopy(nxt[y], 0, foodPher[y], 0, w);
            }
        }
    }

    static final class Ant {
        final World world;
        int x, y;
        boolean carrying;

        Ant(World world) {
            this.world = world;
            this.x = world.nestX; this.y = world.nestY;
            this.carrying = false;
        }

        // Rule 2, fully LOCAL: follow the field that leads where you're going. Searchers read
        // the FOOD scent; carriers read the HOME scent. Brownian floor (1.0) keeps the walk
        // alive even off-trail.
        void step(double depositAmt) {
            double[][] field = carrying ? world.homePher : world.foodPher;
            double[] wts = new double[8];
            double tot = 0.0;
            for (int i = 0; i < 8; i++) {
                int nx = x + DIRS[i][0], ny = y + DIRS[i][1];
                if (!world.free(nx, ny)) {          // rule 1: never step into a wall
                    wts[i] = 0.0;
                } else {
                    wts[i] = 1.0 + field[ny][nx] * 6.0;
                }
                tot += wts[i];
            }
            if (tot <= 0) return;                    // boxed in — stay put this tick

            double r = world.rng.random() * tot;
            double acc = 0.0;
            for (int i = 0; i < 8; i++) {
                acc += wts[i];
                if (r <= acc) {                      // note the <=, matches the reference
                    x += DIRS[i][0]; y += DIRS[i][1];
                    break;
                }
            }
            // rule 3: carriers lay the FOOD trail (searchers lay nothing; the nest emits HOME)
            if (carrying) {
                world.foodPher[y][x] += depositAmt;
            }
            // rule 4: pick up food
            if (world.atFood(x, y) && !carrying && world.foodQty > 0) {
                carrying = true; world.foodQty -= 1;
            } else if (world.atNest(x, y) && carrying) {   // rule 5: drop food at the nest
                carrying = false; world.deliveries += 1;
            }
        }
    }

    static void renderAscii(World world, Ant[] ants) {
        double peak = 0.0;
        for (int y = 0; y < world.h; y++)
            for (int x = 0; x < world.w; x++)
                if (world.foodPher[y][x] > peak) peak = world.foodPher[y][x];
        if (peak == 0.0) peak = 1.0;

        final String shades = " .:-=+*#%@";
        boolean[][] antpos = new boolean[world.h][world.w];
        for (Ant a : ants) antpos[a.y][a.x] = true;

        System.out.println();
        System.out.println("Go to the Ant — emergent trail (nest N <-> food F, wall '|', pheromone density):");
        System.out.println();
        StringBuilder out = new StringBuilder();
        for (int y = 0; y < world.h; y++) {
            StringBuilder line = new StringBuilder();
            for (int x = 0; x < world.w; x++) {
                char c;
                if (x == world.nestX && y == world.nestY) c = 'N';
                else if (x == world.foodX && y == world.foodY) c = 'F';
                else if (world.obstacle[y][x]) c = '|';
                else if (antpos[y][x]) c = 'o';
                else {
                    int lvl = (int) ((world.foodPher[y][x] / peak) * (shades.length() - 1));
                    if (lvl < 0) lvl = 0;
                    if (lvl > shades.length() - 1) lvl = shades.length() - 1;
                    c = shades.charAt(lvl);
                }
                line.append(c);
            }
            out.append(line).append('\n');
        }
        System.out.print(out);
    }

    static void run(int ticks, int nAnts, double evap, double deposit, long seed, boolean useWall, double diffuse) {
        World world = new World(56, 28, seed, useWall, 2);
        Ant[] ants = new Ant[nAnts];
        for (int i = 0; i < nAnts; i++) ants[i] = new Ant(world);

        StringBuilder history = new StringBuilder();
        int step = Math.max(1, ticks / 20);
        for (int t = 0; t < ticks; t++) {
            world.emitAndDiffuseHome();              // nest broadcasts the home gradient
            for (Ant a : ants) a.step(deposit);
            world.diffuseFood(diffuse);              // the food trail spreads a little (breadth)
            world.evaporate(evap);
            if (t % step == 0) {
                if (history.length() > 0) history.append(' ');
                history.append(world.deliveries);
            }
        }

        renderAscii(world, ants);
        System.out.println();
        System.out.println("food delivered to nest over " + ticks + " ticks: " + world.deliveries);
        System.out.println("deliveries(t): " + history);
    }

    public static void main(String[] args) {
        int ticks = 3000, ants = 90;
        double evap = 0.015, deposit = 1.0, diffuse = 0.03;
        long seed = 0;
        boolean useWall = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--ticks": ticks = Integer.parseInt(args[++i]); break;
                case "--ants":  ants  = Integer.parseInt(args[++i]); break;
                case "--evap":  evap  = Double.parseDouble(args[++i]); break;
                case "--diffuse": diffuse = Double.parseDouble(args[++i]); break;
                case "--seed":  seed  = Long.parseLong(args[++i]); break;
                case "--wall":  useWall = true; break;
                default: /* ignore unknown args */ break;
            }
        }
        run(ticks, ants, evap, deposit, seed, useWall, diffuse);
    }
}
