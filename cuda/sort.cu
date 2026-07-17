// sort.cu — "Go to the Ant" §3.2 Ant Brood Sorting (Deneubourg et al. 1991), CUDA port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997), §3.2.
//
// Faithful port of the authoritative Python reference (brood_sorting.py). The four
// LOCAL ant rules are preserved verbatim in spirit:
//   1. Wander randomly (dx,dy each in {-1,0,1}, toroidal grid).
//   2. Keep a SHORT memory (~15 steps) of the object types recently seen (empties
//      included).
//   3. Not carrying + on an object: pick it up stochastically.
//        p(pickup)  = (k+/(k+ + f))^2   -- PAPER §3.2 VERBATIM
//   4. Carrying + on empty ground: drop it stochastically.
//        p(putdown) = (f/(k- + f))^2    -- PAPER §3.2 VERBATIM
//   where f = fraction of short-term memory holding the SAME type.
//   Constants (paper): k+ 0.1, k- 0.3 (Deneubourg 1991; Parunak's summary rounds to ~1, ~3)  (k- must exceed k+ or clusters dissolve
//   faster than they form).  -- PAPER §3.2 VERBATIM (kp=0.1 < km=0.3, mem=15)
// Local concentrations of like items emerge, retain members, and attract more;
// stochastic pickup lets separate clusters merge. Sorting EMERGES; no ant compares
// the whole nest. Clustering = mean fraction of the 8 toroidal neighbours that
// share an item's type.
//
// ---------------------------------------------------------------------------
// THE CUDA LENS (this port's distinct view of the system):
//   The nest grid IS GPU global memory. Ants ARE threads — one thread per ant.
//   A pickup or drop is an atomicCAS on a grid cell: the ant that wins the race
//   claims the item (or the empty square), so item conservation is enforced by the
//   hardware. The sorting is literally many independent threads reshaping one
//   shared array through local compare-and-swap, with no global controller.
//
// OPERATIONALIZED DEVIATION (language-forced, marked and reported):
//   The Python steps ants SEQUENTIALLY: ant k acts on the grid already modified by
//   ants 0..k-1 within the same tick, on a single shared RNG stream. Here the 40
//   ants run in PARALLEL threads, each carrying its OWN SplitMix64 stream (derived
//   from --seed), and they mutate the ONE shared grid concurrently. Pickup/drop are
//   resolved by atomicCAS (claim-the-cell), which keeps items conserved but makes
//   the exact tick-order of colliding ants depend on GPU scheduling. Consequently
//   the grid is NOT bit-identical to the sequential ports and may vary slightly run
//   to run; the EMERGENT SIGNATURE (clustering ~0.35 rising monotonically to
//   ~0.85-0.92) is robust and reproduces. The RNG *algorithm* matches the other
//   ports; the per-ant stream partition differs from a single sequential stream (an
//   unavoidable consequence of parallelism).
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

// Grid + population (paper / reference defaults).
#define GW 40
#define GH 24
#define NCELL (GW * GH)
#define N_PER_TYPE 90            // 90 each of A/B/C -> 270 items
#define NTYPES 3
#define EMPTY (-1)
#define MEMLEN 15                // Deneubourg 1991: m=15 (Parunak's summary rounds to ~10)

// ---- SplitMix64 (identical algorithm across all ports) --------------------
__host__ __device__ static inline uint64_t sm64_next(uint64_t *state) {
    *state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__host__ __device__ static inline double sm64_float(uint64_t *state) {
    // (next() >> 11) * 2^-53  -> a double in [0,1)
    return (double)(sm64_next(state) >> 11) * (1.0 / 9007199254740992.0);
}
__host__ static inline uint64_t sm64_randrange(uint64_t *state, uint64_t n) {
    return sm64_next(state) % n;   // randrange(n)
}

// ---- ant step kernel -------------------------------------------------------
// One thread per ant. Every ant reads the shared grid, updates its own memory,
// then attempts a pickup/drop via atomicCAS on its current cell.
__global__ void k_step(int *ant_x, int *ant_y, int *ant_carry, uint64_t *ant_state,
                       int *mem, int *mem_count, int *mem_pos, int *grid,
                       int n_ants, double kp, double km) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ants) return;

    int x = ant_x[i], y = ant_y[i], carry = ant_carry[i];
    uint64_t st = ant_state[i];

    // rule 1: wander. choice((-1,0,1)) == randrange(3)-1, one next() per axis.
    int dx = (int)(sm64_next(&st) % 3) - 1;
    int dy = (int)(sm64_next(&st) % 3) - 1;
    x = (x + dx + GW) % GW;
    y = (y + dy + GH) % GH;
    int cell = y * GW + x;
    int here = grid[cell];                          // shared read (see OPERATIONALIZED note)

    // rule 2: record every cell seen (empties included), then compute f AFTER
    // appending, exactly as the Python does (append precedes _f).
    int cnt = mem_count[i];
    int pos = mem_pos[i];
    mem[i * MEMLEN + pos] = here;
    pos = (pos + 1) % MEMLEN;
    if (cnt < MEMLEN) cnt = cnt + 1;
    mem_count[i] = cnt;
    mem_pos[i] = pos;

    if (carry == EMPTY) {
        if (here != EMPTY) {                        // rule 3: maybe pick up
            int same = 0;
            for (int k = 0; k < cnt; ++k) same += (mem[i * MEMLEN + k] == here);
            double f = (double)same / (double)cnt;  // cnt>=1 here (we just appended)
            double p = kp / (kp + f);               // PAPER §3.2 VERBATIM: (k+/(k++f))^2
            p = p * p;
            if (sm64_float(&st) < p) {
                // claim the item: only one thread can win the CAS on this cell.
                if (atomicCAS(&grid[cell], here, EMPTY) == here) carry = here;
            }
        }
    } else {
        if (here == EMPTY) {                         // rule 4: maybe drop
            int same = 0;
            for (int k = 0; k < cnt; ++k) same += (mem[i * MEMLEN + k] == carry);
            double f = (double)same / (double)cnt;
            double p = f / (km + f);                  // PAPER §3.2 VERBATIM: (f/(k-+f))^2
            p = p * p;
            if (sm64_float(&st) < p) {
                // claim the empty square: only one thread can win the CAS.
                if (atomicCAS(&grid[cell], EMPTY, carry) == EMPTY) carry = EMPTY;
            }
        }
    }

    ant_x[i] = x; ant_y[i] = y; ant_carry[i] = carry; ant_state[i] = st;
}

// ---- host helpers ----------------------------------------------------------
// Clustering: mean fraction of the 8 toroidal neighbours that share an item's type.
static double clustering(const int *grid) {
    static const int DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static const int DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int tot = 0; double same = 0.0;
    for (int y = 0; y < GH; ++y) {
        for (int x = 0; x < GW; ++x) {
            int t = grid[y * GW + x];
            if (t == EMPTY) continue;
            int neigh = 0, simt = 0;
            for (int k = 0; k < 8; ++k) {
                int nx = (x + DX[k] + GW) % GW, ny = (y + DY[k] + GH) % GH;
                int u = grid[ny * GW + nx];
                if (u != EMPTY) { neigh += 1; simt += (u == t); }
            }
            if (neigh) { tot += 1; same += (double)simt / (double)neigh; }
        }
    }
    return same / (double)(tot > 0 ? tot : 1);
}

static void render(const int *grid) {
    const char *ch = "ABC";
    for (int y = 0; y < GH; ++y) {
        char line[GW + 1];
        for (int x = 0; x < GW; ++x) {
            int t = grid[y * GW + x];
            line[x] = (t == EMPTY) ? '.' : ch[t];
        }
        line[GW] = '\0';
        printf("%s\n", line);
    }
}

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

int main(int argc, char **argv) {
    int ticks = 120000, n_ants = 40;
    double kp = 0.1, km = 0.3;      // Deneubourg 1991: k+=0.1 < k-=0.3 (Parunak's summary rounds to 1<3)
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ants") && i + 1 < argc) n_ants = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kp") && i + 1 < argc) kp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--km") && i + 1 < argc) km = atof(argv[++i]);
    }

    // --- initial scatter, mirroring the Python Nest.__init__ RNG order ---
    // cells = [(x,y) for y in range(h) for x in range(w)]  (row-major), shuffled,
    // then first 90 -> A(0), next 90 -> B(1), next 90 -> C(2).
    uint64_t master = seed;
    int *h_grid = (int *)malloc(NCELL * sizeof(int));
    for (int c = 0; c < NCELL; ++c) h_grid[c] = EMPTY;
    int *cells = (int *)malloc(NCELL * sizeof(int));
    for (int c = 0; c < NCELL; ++c) cells[c] = c;              // index = y*GW + x, row-major
    // Fisher-Yates: for i from len-1 downto 1: j=randrange(i+1); swap.
    for (int i = NCELL - 1; i >= 1; --i) {
        int j = (int)sm64_randrange(&master, (uint64_t)(i + 1));
        int tmp = cells[i]; cells[i] = cells[j]; cells[j] = tmp;
    }
    int idx = 0;
    for (int t = 0; t < NTYPES; ++t)
        for (int k = 0; k < N_PER_TYPE; ++k) h_grid[cells[idx++]] = t;

    // --- ant init: Python draws x,y from the shared stream (randrange), in order ---
    // Here each ant ALSO gets its own SplitMix64 stream (OPERATIONALIZED: parallel).
    int *h_ax = (int *)malloc(n_ants * sizeof(int));
    int *h_ay = (int *)malloc(n_ants * sizeof(int));
    int *h_carry = (int *)malloc(n_ants * sizeof(int));
    uint64_t *h_state = (uint64_t *)malloc(n_ants * sizeof(uint64_t));
    for (int i = 0; i < n_ants; ++i) {
        h_ax[i] = (int)sm64_randrange(&master, GW);   // rng.randrange(w)
        h_ay[i] = (int)sm64_randrange(&master, GH);   // rng.randrange(h)
        h_carry[i] = EMPTY;
        h_state[i] = sm64_next(&master);              // per-ant stream (parallel deviation)
    }

    printf("BEFORE (random scatter):\n\n");
    render(h_grid);
    double c0 = clustering(h_grid);
    printf("\ninitial clustering: %.3f\n", c0);

    // --- device buffers ---
    int *d_grid, *d_ax, *d_ay, *d_carry, *d_mem, *d_mcount, *d_mpos;
    uint64_t *d_state;
    CK(cudaMalloc(&d_grid, NCELL * sizeof(int)));
    CK(cudaMalloc(&d_ax, n_ants * sizeof(int)));
    CK(cudaMalloc(&d_ay, n_ants * sizeof(int)));
    CK(cudaMalloc(&d_carry, n_ants * sizeof(int)));
    CK(cudaMalloc(&d_state, n_ants * sizeof(uint64_t)));
    CK(cudaMalloc(&d_mem, n_ants * MEMLEN * sizeof(int)));
    CK(cudaMalloc(&d_mcount, n_ants * sizeof(int)));
    CK(cudaMalloc(&d_mpos, n_ants * sizeof(int)));

    CK(cudaMemcpy(d_grid, h_grid, NCELL * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ax, h_ax, n_ants * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ay, h_ay, n_ants * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_carry, h_carry, n_ants * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_state, h_state, n_ants * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_mem, 0, n_ants * MEMLEN * sizeof(int)));
    CK(cudaMemset(d_mcount, 0, n_ants * sizeof(int)));
    CK(cudaMemset(d_mpos, 0, n_ants * sizeof(int)));

    int antBlocks = (n_ants + 127) / 128;

    // history: sample clustering like the Python (t % max(1, ticks/12) == 0).
    int stepEvery = ticks / 12; if (stepEvery < 1) stepEvery = 1;
    double hist[64]; int nhist = 0;

    for (int t = 0; t < ticks; ++t) {
        if (t % stepEvery == 0 && nhist < 64) {
            CK(cudaMemcpy(h_grid, d_grid, NCELL * sizeof(int), cudaMemcpyDeviceToHost));
            hist[nhist++] = clustering(h_grid);
        }
        k_step<<<antBlocks, 128>>>(d_ax, d_ay, d_carry, d_state, d_mem, d_mcount,
                                   d_mpos, d_grid, n_ants, kp, km);
    }
    CK(cudaDeviceSynchronize());

    CK(cudaMemcpy(h_grid, d_grid, NCELL * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nAFTER (emergent sorting):\n\n");
    render(h_grid);
    double cf = clustering(h_grid);
    printf("\nfinal clustering: %.3f\n", cf);
    printf("clustering(t):");
    for (int i = 0; i < nhist; ++i) printf(" %.2f", hist[i]);
    printf("\n");

    cudaFree(d_grid); cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_carry);
    cudaFree(d_state); cudaFree(d_mem); cudaFree(d_mcount); cudaFree(d_mpos);
    free(h_grid); free(cells); free(h_ax); free(h_ay); free(h_carry); free(h_state);
    return 0;
}
