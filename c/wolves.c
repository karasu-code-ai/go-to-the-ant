/* "Go to the Ant" §3.6 — Wolves: Surrounding Prey (Korf 1992), C port.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
 * Underlying model: R. E. Korf, "A simple solution to pursuit games" (1992).
 *
 * THE LENS (C): the state is a raw array of doubles; the mechanism laid utterly
 * bare. Wolves are a flat float array, the field of candidate moves is generated
 * by hand, and every distance is a hypot in a manual loop — no abstraction
 * between the agent and the geometry. Manual memory, hand-rolled RNG.
 *
 * Two local rules (Parunak §3.6):
 *   1. MOOSE: move to the neighbouring candidate FARTHEST from its nearest wolf.
 *   2. WOLVES: move to minimise  S = d(moose) - k*d(nearest other wolf)  — get
 *      CLOSE to the moose while staying FAR from each other. k tunes repulsion.
 * Attraction (to prey) balanced against repulsion (between wolves) makes the
 * pack encircle the moose with no communication.
 *
 * PROVENANCE:
 *   VERBATIM (paper): the score  S = d(moose) - k*d(wolf).
 *   OPERATIONALIZED: continuous plane + a 24-candidate-direction search (plus
 *     staying put) instead of the paper's hex grid; speeds vm=0.6, vw=1.0,
 *     k=1.12. The paper states six wolves capture on a hex grid; here a
 *     continuous plane with a candidate-direction search.
 * No per-step randomness — deterministic given the initial wolf placement.
 *
 * Dependency-free: C standard library only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define TAU (2.0 * M_PI)

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
static inline double rng_uniform(double a, double b) { /* a + random_float()*(b-a) */
    return a + rng_float() * (b - a);
}

/* --- The world. --- */
#define WORLD_W 80
#define WORLD_H 44
#define VM 0.6
#define VW 1.0
#define KREP 1.12
#define NCAND 25   /* 24 directions + staying put */

static double wolf_x[64], wolf_y[64];
static double moose_x, moose_y;

static inline double hyp(double dx, double dy) { return sqrt(dx * dx + dy * dy); }
static inline int in_bounds(double x, double y) {
    return x >= 0.0 && x < (double)WORLD_W && y >= 0.0 && y < (double)WORLD_H;
}

/* The 25 candidate moves for an agent at (x,y) with speed sp: 24 points around
 * the compass plus staying put — exactly the Python _cands order. */
static void candidates(double x, double y, double sp, double *cx, double *cy) {
    for (int i = 0; i < 24; i++) {
        double a = (double)i * TAU / 24.0;
        cx[i] = x + sp * cos(a);
        cy[i] = y + sp * sin(a);
    }
    cx[24] = x; cy[24] = y;
}

/* One tick. rule 1 (moose) then rule 2 (wolves), matching Python order. */
static void step(int n) {
    double cx[NCAND], cy[NCAND];

    /* rule 1: moose flees to the candidate farthest from its nearest wolf */
    candidates(moose_x, moose_y, VM, cx, cy);
    double bestx = moose_x, besty = moose_y, bd = -1.0;
    for (int c = 0; c < NCAND; c++) {
        if (!in_bounds(cx[c], cy[c])) continue;
        double dmin = 1e18;
        for (int j = 0; j < n; j++) {
            double d = hyp(cx[c] - wolf_x[j], cy[c] - wolf_y[j]);
            if (d < dmin) dmin = d;
        }
        if (dmin > bd) { bd = dmin; bestx = cx[c]; besty = cy[c]; }
    }
    moose_x = bestx; moose_y = besty;

    /* rule 2: each wolf minimises S = d(moose) - k*d(nearest OTHER wolf).
     * OPERATIONALIZED: wolves are updated in index order using the ALREADY
     * moved moose but the CURRENT (pre-move) positions of the other wolves —
     * this is exactly the Python's sequential-read-from-W-into-nw pattern. */
    double nx[64], ny[64];
    for (int i = 0; i < n; i++) {
        candidates(wolf_x[i], wolf_y[i], VW, cx, cy);
        double bx = wolf_x[i], by = wolf_y[i], bs = 1e9;
        for (int c = 0; c < NCAND; c++) {
            if (!in_bounds(cx[c], cy[c])) continue;
            double dm = hyp(cx[c] - moose_x, cy[c] - moose_y);
            double doo = 0.0; int have = 0;
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                double d = hyp(cx[c] - wolf_x[j], cy[c] - wolf_y[j]);
                if (!have || d < doo) { doo = d; have = 1; }
            }
            double s = dm - KREP * doo;         /* VERBATIM: S = d(moose) - k*d(wolf) */
            if (s < bs) { bs = s; bx = cx[c]; by = cy[c]; }
        }
        nx[i] = bx; ny[i] = by;
    }
    for (int i = 0; i < n; i++) { wolf_x[i] = nx[i]; wolf_y[i] = ny[i]; }
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Largest angular gap (deg) between adjacent wolves as seen from the moose.
 * 360/N when evenly ringed -> surrounded; near 360 when all on one side. */
static double gap(int n) {
    if (n < 2) return 360.0;
    double angs[64];
    for (int i = 0; i < n; i++)
        angs[i] = atan2(wolf_y[i] - moose_y, wolf_x[i] - moose_x);
    qsort(angs, n, sizeof(double), cmp_double);
    double mx = 0.0;
    for (int i = 0; i < n; i++) {
        double g = angs[(i + 1) % n] - angs[i];
        g = fmod(g, TAU);
        if (g < 0) g += TAU;
        if (g > mx) mx = g;
    }
    return mx * 180.0 / M_PI;
}

static double nearest_wolf(int n) {
    double md = 1e18;
    for (int i = 0; i < n; i++) {
        double d = hyp(moose_x - wolf_x[i], moose_y - wolf_y[i]);
        if (d < md) md = d;
    }
    return md;
}

static void render_ascii(int n) {
    static char grid[WORLD_H][WORLD_W];
    memset(grid, ' ', sizeof grid);
    for (int i = 0; i < n; i++) {
        int x = ((int)wolf_x[i]) % WORLD_W, y = ((int)wolf_y[i]) % WORLD_H;
        if (x < 0) x += WORLD_W;
        if (y < 0) y += WORLD_H;
        grid[y][x] = 'W';
    }
    int mx = ((int)moose_x) % WORLD_W, my = ((int)moose_y) % WORLD_H;
    if (mx < 0) mx += WORLD_W;
    if (my < 0) my += WORLD_H;
    grid[my][mx] = 'M';
    printf("\nThe hunt (M moose, W wolves - watch the ring close):\n\n");
    for (int y = 0; y < WORLD_H; y++) {
        for (int x = 0; x < WORLD_W; x++) putchar(grid[y][x]);
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    long ticks = 260, n_wolves = 6;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--wolves") && i + 1 < argc) n_wolves = strtol(argv[++i], NULL, 10);
    }
    if (n_wolves > 64) n_wolves = 64;
    rng_state = seed;

    moose_x = (double)WORLD_W / 2.0;
    moose_y = (double)WORLD_H / 2.0;
    /* wolves uniform-random; RNG order: uniform(0,w) then uniform(0,h) per wolf */
    for (int i = 0; i < n_wolves; i++) {
        wolf_x[i] = rng_uniform(0.0, (double)WORLD_W);
        wolf_y[i] = rng_uniform(0.0, (double)WORLD_H);
    }

    double hist[64]; int nhist = 0;
    hist[nhist++] = gap((int)n_wolves);
    long every = ticks / 12; if (every < 1) every = 1;
    for (long t = 0; t < ticks; t++) {
        step((int)n_wolves);
        if (t % every == 0 && nhist < 64) hist[nhist++] = gap((int)n_wolves);
    }

    render_ascii((int)n_wolves);
    printf("\nlargest escape gap around the moose: %.0f deg  "
           "(evenly surrounded ~ %ld deg) | nearest wolf %.1f\n",
           gap((int)n_wolves), 360 / n_wolves, nearest_wolf((int)n_wolves));
    printf("gap deg(t):");
    for (int i = 0; i < nhist; i++) printf(" %.0f", hist[i]);
    printf("\n");
    return 0;
}
