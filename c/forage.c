/* "Go to the Ant" — foraging swarm, C port.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
 *
 * THE LENS (C): the pheromone field is a raw shared array of doubles; stigmergy
 * laid utterly bare. Manual memory, hand-rolled RNG, index arithmetic on a flat
 * grid — the hardcore baseline against which every other port is measured.
 *
 * The five local ant rules (§3.1 "Ants: Path planning"), operationalized:
 *   1. Avoid obstacles (never step into a non-free cell).
 *   2. Wander randomly, biased toward nearby pheromone (Brownian floor of 1.0
 *      plus a local scent bias; no pheromone -> uniform random walk).
 *   3. If carrying food, drop pheromone at a CONSTANT RATE while walking.
 *   4. At food and empty-handed -> pick it up.
 *   5. At the nest carrying food -> drop it (a delivery).
 * Plus the field law: pheromone EVAPORATES every tick (the entropy leak) so
 * dead trails fade. No ant plans a route; the trail EMERGES from deposit +
 * evaporation + weighted-random following.
 *
 * TWO local pheromone fields — a principled "communication through the
 * environment" (§4.3.3 / §4.6), NOT a global homing beacon:
 *   food_pher: laid by CARRIERS, followed by SEARCHERS.
 *   home_pher: emitted+diffused by the NEST, followed by CARRIERS.
 * No ant knows where the nest is; a carrier just climbs the local home gradient.
 *
 * Dependency-free: C standard library only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define W 56
#define H 28
#define IDX(x, y) ((y) * W + (x))

/* --- SplitMix64 PRNG (identical across all ports, seeded from --seed) --- */
static uint64_t rng_state;
static inline uint64_t rng_next(void) {
    rng_state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double rng_float(void) { /* [0,1) */
    return (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0); /* 2^-53 */
}

/* The 8 neighbours, in THIS order (matches Python DIRS). */
static const int DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
static const int DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

/* --- The world: raw shared arrays. --- */
static double food_pher[W * H];   /* laid by carriers; followed by searchers */
static double home_pher[W * H];   /* emitted+diffused by nest; followed by carriers */
static int    obstacle[W * H];
static double home_tmp[W * H];    /* Jacobi scratch for simultaneous diffusion */

static const int NEST_X = 5,  NEST_Y = H / 2;   /* (5, 14) */
static const int FOOD_X = W - 6, FOOD_Y = H / 2; /* (50, 14) */
static const int REGION = 2;
static long long food_qty = 1000000000LL;
static long long deliveries = 0;

typedef struct { int x, y, carrying; } Ant;

static inline int in_bounds(int x, int y) {
    return x >= 0 && x < W && y >= 0 && y < H;
}
static inline int is_free(int x, int y) {
    return in_bounds(x, y) && !obstacle[IDX(x, y)];
}
static inline int at_region(int x, int y, int cx, int cy) {
    int ax = x - cx, ay = y - cy;
    if (ax < 0) ax = -ax;
    if (ay < 0) ay = -ay;
    return ax <= REGION && ay <= REGION;
}

/* Step 1: the NEST is a home-pheromone SOURCE; the marker diffuses outward into
 * a gradient that points home from everywhere. Carriers read only the LOCAL
 * gradient — no global nest-direction. Jacobi (simultaneous) update. */
static void emit_and_diffuse_home(void) {
    for (int dy = -REGION; dy <= REGION; dy++)
        for (int dx = -REGION; dx <= REGION; dx++) {
            int x = NEST_X + dx, y = NEST_Y + dy;
            if (is_free(x, y)) home_pher[IDX(x, y)] += 6.0;
        }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int i = IDX(x, y);
            if (!is_free(x, y)) { home_tmp[i] = home_pher[i]; continue; }
            double s = home_pher[i];
            int c = 1;
            for (int d = 0; d < 8; d++) {
                int xx = x + DX[d], yy = y + DY[d];
                if (is_free(xx, yy)) { s += home_pher[IDX(xx, yy)]; c++; }
            }
            home_tmp[i] = s / c;
        }
    memcpy(home_pher, home_tmp, sizeof home_pher);
}

/* Step 2: one ant's move. Rule 2 fully local: follow the field that leads where
 * you're going (home if carrying, food if searching). Weighted-random pick over
 * the 8 dirs, then rules 3/4/5. */
static void step_ant(Ant *a, double deposit) {
    const double *field = a->carrying ? home_pher : food_pher;
    double wts[8], tot = 0.0;
    for (int d = 0; d < 8; d++) {
        int nx = a->x + DX[d], ny = a->y + DY[d];
        if (!is_free(nx, ny)) { wts[d] = 0.0; continue; } /* rule 1 */
        wts[d] = 1.0 + field[IDX(nx, ny)] * 6.0;          /* Brownian floor + scent */
        tot += wts[d];
    }
    if (tot <= 0.0) return;                               /* boxed in — stay put */
    double r = rng_float() * tot, acc = 0.0;
    for (int d = 0; d < 8; d++) {
        acc += wts[d];
        if (r <= acc) { a->x += DX[d]; a->y += DY[d]; break; } /* note the <= */
    }
    /* rule 3: carriers lay the FOOD trail (nest broadcasts HOME; searchers lay nothing) */
    if (a->carrying) food_pher[IDX(a->x, a->y)] += deposit;
    /* rule 4: pick up food */
    if (at_region(a->x, a->y, FOOD_X, FOOD_Y) && !a->carrying && food_qty > 0) {
        a->carrying = 1; food_qty--;
    } else if (at_region(a->x, a->y, NEST_X, NEST_Y) && a->carrying) {
        /* rule 5: drop food at the nest */
        a->carrying = 0; deliveries++;
    }
}

/* Step 3: both fields dissipate every tick (the entropy leak). */
static void evaporate(double rate) {
    double keep = 1.0 - rate;
    for (int i = 0; i < W * H; i++) { food_pher[i] *= keep; home_pher[i] *= keep; }
}

static void render_ascii(const Ant *ants, int n) {
    double peak = 0.0;
    for (int i = 0; i < W * H; i++) if (food_pher[i] > peak) peak = food_pher[i];
    if (peak <= 0.0) peak = 1.0;
    const char *shades = " .:-=+*#%@";
    int nsh = 10;
    /* mark ant positions */
    static int antpos[W * H];
    memset(antpos, 0, sizeof antpos);
    for (int k = 0; k < n; k++) antpos[IDX(ants[k].x, ants[k].y)] = 1;
    printf("\nGo to the Ant — emergent trail (nest N <-> food F, wall '|', pheromone density):\n\n");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            char c;
            if (x == NEST_X && y == NEST_Y) c = 'N';
            else if (x == FOOD_X && y == FOOD_Y) c = 'F';
            else if (obstacle[IDX(x, y)]) c = '|';
            else if (antpos[IDX(x, y)]) c = 'o';
            else {
                int lvl = (int)((food_pher[IDX(x, y)] / peak) * (nsh - 1));
                if (lvl < 0) lvl = 0;
                if (lvl > nsh - 1) lvl = nsh - 1;
                c = shades[lvl];
            }
            putchar(c);
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    long ticks = 3000, n_ants = 90;
    double evap = 0.015, deposit = 1.0;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ants") && i + 1 < argc) n_ants = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--evap") && i + 1 < argc) evap = strtod(argv[++i], NULL);
    }
    rng_state = seed;

    memset(food_pher, 0, sizeof food_pher);
    memset(home_pher, 0, sizeof home_pher);
    memset(obstacle, 0, sizeof obstacle);

    Ant *ants = (Ant *)malloc(sizeof(Ant) * (size_t)n_ants);
    for (long k = 0; k < n_ants; k++) { ants[k].x = NEST_X; ants[k].y = NEST_Y; ants[k].carrying = 0; }

    /* S-curve sample: 20 evenly-spaced cumulative-delivery readings. */
    long step = ticks / 20; if (step < 1) step = 1;
    long hist[64]; int nhist = 0;
    for (long t = 0; t < ticks; t++) {
        emit_and_diffuse_home();                 /* nest broadcasts the home gradient */
        for (long k = 0; k < n_ants; k++) step_ant(&ants[k], deposit);
        evaporate(evap);                         /* both fields dissipate */
        if (t % step == 0 && nhist < 64) hist[nhist++] = deliveries;
    }

    render_ascii(ants, (int)n_ants);
    printf("\nfood delivered to nest over %ld ticks: %lld\n", ticks, deliveries);
    printf("deliveries(t):");
    for (int i = 0; i < nhist; i++) printf(" %ld", hist[i]);
    printf("\n");

    free(ants);
    return 0;
}
