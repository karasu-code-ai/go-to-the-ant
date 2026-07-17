/* "Go to the Ant" — ant brood sorting, C port.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
 * §3.2 (after Deneubourg et al. 1991).
 *
 * THE LENS (C): the nest is a raw array of chars — one byte per cell, 0 for
 * empty ground or a type tag A/B/C. No object, no abstraction between the agent
 * and the field it edits: an ant is three ints (x, y, carry) plus a ring buffer
 * of the last 10 cells it saw. Pickup and putdown are direct byte writes into
 * the shared grid. The sorting mechanism laid utterly bare.
 *
 * The four local rules (§3.2), operationalized from the paper:
 *   1. Wander randomly around the nest (dx,dy each in {-1,0,1}, toroidal).
 *   2. Keep a SHORT memory (~15 steps) of the object types recently seen
 *      (record every cell, including empties).                  [OPERATIONALIZED:
 *      "short memory ~10" is qualitative in the paper; we use mem=15 (Deneubourg 1991).]
 *   3. Not carrying + at an object: pick it up stochastically with
 *          p(pickup) = (k+ / (k+ + f))^2                        [PAPER §3.2 VERBATIM]
 *      where f is the fraction of memory holding the SAME type.
 *   4. Carrying + on empty ground: drop it stochastically with
 *          p(putdown) = (f / (k- + f))^2                        [PAPER §3.2 VERBATIM]
 *   Constants (paper): k+ 0.1, k- 0.3 (Deneubourg 1991; Parunak's summary rounds to ~1, ~3) (k- must exceed k+ or clusters dissolve
 *   faster than they form).                                     [PAPER §3.2]
 * Local concentrations of like items emerge, retain members, and attract more;
 * stochastic pickup lets separate clusters merge. Sorting EMERGES; no ant
 * compares the whole nest.
 *
 * Dependency-free: C standard library only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define W 40
#define H 24
#define IDX(x, y) ((y) * W + (x))
#define N_PER_TYPE 90     /* 90 each of A/B/C -> 270 items scattered */
#define MEM 15            /* rule 2: short memory (Deneubourg 1991: m=15; Parunak rounds to ~10) */

/* --- SplitMix64 PRNG (identical across all ports, seeded from --seed).
 * A CROSS-PORT CONVENTION; intentionally NOT CPython's random module. All
 * sequential ports consume the stream in the same order, so they agree with
 * each other. --- */
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

/* --- The nest: a raw shared array. 0 = empty ground; 1/2/3 = type A/B/C. --- */
static unsigned char grid[W * H];

static const char TYPES[3] = {'A', 'B', 'C'};

/* rule 1 move offsets: choice((-1,0,1)) == CHOICE[randrange(3)]. */
static const int CHOICE[3] = {-1, 0, 1};

/* Scatter N_PER_TYPE of each type across shuffled cells (mirrors Nest.__init__:
 * build the cell list in row-major order, Fisher-Yates shuffle, fill in order). */
static void scatter(void) {
    int cells[W * H][2];
    int m = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) { cells[m][0] = x; cells[m][1] = y; m++; }
    /* shuffle(cells): Fisher-Yates, i from len-1 downto 1, j=randrange(i+1). */
    for (int i = m - 1; i >= 1; i--) {
        int j = (int)rng_randrange((uint64_t)(i + 1));
        int tx = cells[i][0], ty = cells[i][1];
        cells[i][0] = cells[j][0]; cells[i][1] = cells[j][1];
        cells[j][0] = tx;          cells[j][1] = ty;
    }
    memset(grid, 0, sizeof grid);
    int idx = 0;
    for (int t = 0; t < 3; t++)
        for (int k = 0; k < N_PER_TYPE; k++) {
            int x = cells[idx][0], y = cells[idx][1]; idx++;
            grid[IDX(x, y)] = (unsigned char)(t + 1);   /* tag 1/2/3 */
        }
}

/* Clustering: mean fraction of the 8 toroidal neighbours that share an item's
 * type — 0 = scattered, 1 = perfectly sorted. */
static double clustering(void) {
    long tot = 0;
    double same = 0.0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            unsigned char t = grid[IDX(x, y)];
            if (t == 0) continue;
            int neigh = 0, simt = 0;
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = ((x + dx) % W + W) % W;
                    int ny = ((y + dy) % H + H) % H;
                    unsigned char u = grid[IDX(nx, ny)];
                    if (u != 0) { neigh++; if (u == t) simt++; }
                }
            if (neigh) { tot++; same += (double)simt / neigh; }
        }
    return same / (tot > 0 ? tot : 1);
}

/* --- One ant: position, what it carries (0/1/2/3), and a ring-buffer memory of
 * the last MEM cells seen (each 0/1/2/3). --- */
typedef struct {
    int x, y;
    unsigned char carry;
    unsigned char mem[MEM];
    int mem_len, mem_head;
} Ant;

/* rule 2/3/4 helper: fraction of memory holding type t. Order within the ring
 * doesn't matter — only the count. Empty (0) entries never match a type. */
static double mem_frac(const Ant *a, unsigned char t) {
    if (a->mem_len == 0) return 0.0;
    int c = 0;
    for (int i = 0; i < a->mem_len; i++) if (a->mem[i] == t) c++;
    return (double)c / a->mem_len;
}

static void step_ant(Ant *a) {
    /* rule 1: wander (dx then dy, matching Python's two choice() calls). */
    a->x = ((a->x + CHOICE[rng_randrange(3)]) % W + W) % W;
    a->y = ((a->y + CHOICE[rng_randrange(3)]) % H + H) % H;
    unsigned char here = grid[IDX(a->x, a->y)];
    /* rule 2: record even empties into short memory. */
    a->mem[a->mem_head] = here;
    a->mem_head = (a->mem_head + 1) % MEM;
    if (a->mem_len < MEM) a->mem_len++;

    if (a->carry == 0) {
        if (here != 0) {                                  /* rule 3: maybe pick up */
            double f = mem_frac(a, here);
            double p = 0.1 / (0.1 + f);                    /* kp=0.1 -> kp/(kp+f) */
            p = p * p;                                     /* PAPER §3.2 VERBATIM: (k+/(k++f))^2 */
            if (rng_float() < p) { a->carry = here; grid[IDX(a->x, a->y)] = 0; }
        }
    } else {
        if (here == 0) {                                  /* rule 4: maybe drop */
            double f = mem_frac(a, a->carry);
            double p = f / (0.3 + f);                       /* km=0.3 -> f/(km+f) */
            p = p * p;                                      /* PAPER §3.2 VERBATIM: (f/(k-+f))^2 */
            if (rng_float() < p) { grid[IDX(a->x, a->y)] = a->carry; a->carry = 0; }
        }
    }
}

static void render(void) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            unsigned char t = grid[IDX(x, y)];
            putchar(t == 0 ? '.' : TYPES[t - 1]);
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    long ticks = 120000, n_ants = 40;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ants") && i + 1 < argc) n_ants = strtol(argv[++i], NULL, 10);
    }
    rng_state = seed;

    scatter();

    /* Create ants AFTER the scatter shuffle, in order — each draws x then y
     * (mirrors SortAnt.__init__: randrange(w) then randrange(h)). */
    Ant *ants = (Ant *)calloc((size_t)n_ants, sizeof(Ant));
    for (long k = 0; k < n_ants; k++) {
        ants[k].x = (int)rng_randrange(W);
        ants[k].y = (int)rng_randrange(H);
        ants[k].carry = 0;
        ants[k].mem_len = 0;
        ants[k].mem_head = 0;
    }

    printf("BEFORE (random scatter):\n\n");
    render();
    printf("\ninitial clustering: %.3f\n", clustering());

    long sample = ticks / 12; if (sample < 1) sample = 1;
    double hist[64]; int nhist = 0;
    for (long t = 0; t < ticks; t++) {
        for (long k = 0; k < n_ants; k++) step_ant(&ants[k]);
        if (t % sample == 0 && nhist < 64) hist[nhist++] = clustering();
    }

    printf("\nAFTER (emergent sorting):\n\n");
    render();
    printf("\nfinal clustering: %.3f\n", clustering());
    printf("clustering(t):");
    for (int i = 0; i < nhist; i++) printf(" %.2f", hist[i]);
    printf("\n");

    free(ants);
    return 0;
}
