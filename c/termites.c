/* "Go to the Ant" — termite nest building, C port.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
 * §3.3 (Termite nest building, after Kugler/Turvey; Kugler et al. 1990).
 *
 * THE LENS (C): the state is a raw array; the mechanism laid utterly bare with
 * manual memory, no abstraction between agent and field. TWO flat fields of
 * doubles — mass[] (persistent structure, what you see) and scent[] (decaying
 * pheromone, what biases wandering) — plus a struct-of-scalars termite. Index
 * arithmetic on a toroidal grid; hand-rolled SplitMix64. Nothing between the
 * agent's load and the cell it drops it in.
 *
 * Parunak's three local rules (§3.3):
 *   1. Metabolize bodily waste, which contains pheromone. The waste IS the
 *      building material: load = min(maxload, load + metab).
 *   2. Wander randomly, but prefer the direction of the strongest local
 *      pheromone: over the 8 toroidal neighbours, weight = 1 + scent*3.
 *   3. Each step, stochastically decide whether to deposit the current load.
 *      p(deposit) rises with LOCAL pheromone density AND load carried; a full
 *      termite always drops. On deposit: mass += load, scent += load, load = 0.
 * Because scent DECAYS every tick, the freshest deposits (the centre of a
 * growing pile) smell strongest, so piles climb into COLUMNS rather than
 * spreading. No termite plans the mound.
 *
 * Emergence = scattered dabs self-concentrate into a HANDFUL of tall columns.
 *
 * Dependency-free: C standard library only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define W 58
#define H 34
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
static inline uint64_t rng_randrange(uint64_t n) { return rng_next() % n; }

/* The 8 neighbours, in THIS order (matches Python DIRS). */
static const int DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
static const int DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

/* --- The world: raw shared arrays. --- */
static double mass[W * H];   /* persistent structural mass (viz) */
static double scent[W * H];  /* decaying pheromone (bias) */

typedef struct { int x, y; double load; } Termite;

static double g_metab = 0.4, g_maxload = 6.0;

/* mod that handles the small negative offsets from DX/DY (always > -W, > -H). */
static inline int wrap(int v, int n) { return (v + n) % n; }

/* One termite's step: metabolize, wander (scent-biased), stochastic deposit. */
static void step_termite(Termite *t) {
    /* rule 1: metabolize -> waste accumulates */
    t->load = t->load + g_metab;
    if (t->load > g_maxload) t->load = g_maxload;

    /* rule 2: wander, biased toward the strongest local scent */
    double wts[8], tot = 0.0;
    int nxs[8], nys[8];
    for (int d = 0; d < 8; d++) {
        int nx = wrap(t->x + DX[d], W), ny = wrap(t->y + DY[d], H);
        nxs[d] = nx; nys[d] = ny;
        double w = 1.0 + scent[IDX(nx, ny)] * 3.0;
        wts[d] = w; tot += w;
    }
    double r = rng_float() * tot;
    for (int d = 0; d < 8; d++) {
        r -= wts[d];
        if (r <= 0.0) { t->x = nxs[d]; t->y = nys[d]; break; }
    }

    /* rule 3: stochastic deposit — rises with local scent AND load; full termite
     * always drops. OPERATIONALIZED: paper §3.3 gives NO formula, only "prob
     * rises with local density AND load". */
    double local = scent[IDX(t->x, t->y)];
    double p = 0.01 + 0.55 * (t->load / g_maxload) + 0.20 * local;
    if (p > 1.0) p = 1.0;
    if (t->load >= g_maxload || rng_float() < p) {
        int i = IDX(t->x, t->y);
        mass[i] += t->load;
        scent[i] += t->load;
        t->load = 0.0;
    }
}

/* The field law: scent evaporates every tick (the entropy leak). */
static void evaporate(double decay) {
    double keep = 1.0 - decay;
    for (int i = 0; i < W * H; i++) scent[i] *= keep;
}

/* columns() = count of toroidal local maxima with mass above 0.15*peak. */
static int columns(double *peak_out) {
    double peak = 0.0;
    for (int i = 0; i < W * H; i++) if (mass[i] > peak) peak = mass[i];
    *peak_out = peak;
    if (peak <= 0.0) return 0;
    double cut = peak * 0.15;
    int cnt = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            double v = mass[IDX(x, y)];
            if (v < cut) continue;
            int is_max = 1;
            for (int d = 0; d < 8; d++) {
                int nx = wrap(x + DX[d], W), ny = wrap(y + DY[d], H);
                if (v < mass[IDX(nx, ny)]) { is_max = 0; break; }
            }
            if (is_max) cnt++;
        }
    return cnt;
}

static void render_ascii(void) {
    double peak = 0.0;
    for (int i = 0; i < W * H; i++) if (mass[i] > peak) peak = mass[i];
    if (peak <= 0.0) peak = 1.0;
    const char *shades = " .:-=+*#%@";
    int nsh = 10;
    printf("\nTermite mound (top-down mass density — columns emerge as bright cores):\n\n");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int lvl = (int)(mass[IDX(x, y)] / peak * (nsh - 1));
            if (lvl < 0) lvl = 0;
            if (lvl > nsh - 1) lvl = nsh - 1;
            putchar(shades[lvl]);
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    long ticks = 40000, n = 70;
    double decay = 0.02;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--termites") && i + 1 < argc) n = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ants") && i + 1 < argc) n = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--decay") && i + 1 < argc) decay = strtod(argv[++i], NULL);
    }
    rng_state = seed;

    memset(mass, 0, sizeof mass);
    memset(scent, 0, sizeof scent);

    /* Termite ctor consumes the RNG as Python does: randrange(w) then randrange(h)
     * per termite, in creation order — keeps the stream bit-identical across ports. */
    Termite *tm = (Termite *)malloc(sizeof(Termite) * (size_t)n);
    for (long k = 0; k < n; k++) {
        tm[k].x = (int)rng_randrange(W);
        tm[k].y = (int)rng_randrange(H);
        tm[k].load = 0.0;
    }

    /* Sample columns(t) at the same cadence as Python: every ticks//12 ticks. */
    long stepmod = ticks / 12; if (stepmod < 1) stepmod = 1;
    int hist[64]; int nhist = 0;
    double peak;
    for (long t = 0; t < ticks; t++) {
        for (long k = 0; k < n; k++) step_termite(&tm[k]);
        evaporate(decay);
        if (t % stepmod == 0 && nhist < 64) hist[nhist++] = columns(&peak);
    }

    render_ascii();
    int cnt = columns(&peak);
    printf("\ndistinct columns (local maxima): %d | tallest column mass: %.0f\n", cnt, peak);
    printf("columns(t):");
    for (int i = 0; i < nhist; i++) printf(" %d", hist[i]);
    printf("\n");

    free(tm);
    return 0;
}
