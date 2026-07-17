// forage.cu — "Go to the Ant" foraging swarm, CUDA port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997), §3.1.
//
// Faithful port of the authoritative Python reference (go_to_the_ant.py). The
// five LOCAL ant rules are preserved verbatim in spirit:
//   1. Avoid obstacles (never step into a wall / out of bounds).
//   2. Wander randomly, biased toward the local pheromone scent (Brownian floor
//      of 1.0 per free direction + field*6.0 bias).
//   3. Carriers drop FOOD pheromone at a constant rate while walking.
//   4. At food and empty-handed -> pick up.
//   5. At the nest and carrying -> drop (a delivery).
// Two LOCAL fields (principled "communication through the environment", §4.3.3/§4.6):
//   food_pher: laid by CARRIERS,  followed by SEARCHERS.
//   home_pher: emitted+diffused by the NEST, followed by CARRIERS.
// No ant knows where the nest is — carriers just climb the local home gradient.
// Evaporation every tick is the entropy leak that lets stale trails fade so the
// network (a min spanning tree, Goss et al. 1989) EMERGES rather than being planned.
//
// ---------------------------------------------------------------------------
// THE CUDA LENS (this port's distinct view of the system):
//   The pheromone field IS GPU global memory. Ants ARE threads. Deposits ARE
//   atomicAdd race-resolution — many ants writing the same cell in the same
//   instant is exactly the stigmergic superposition the paper describes, made
//   literal by the hardware.
//
// OPERATIONALIZED DEVIATION (language-forced, marked and reported):
//   The Python steps ants SEQUENTIALLY: ant k sees the fresh deposits of ants
//   0..k-1 laid earlier in the same tick. Here ants run in PARALLEL threads, so
//   every ant reads the SAME pre-tick field snapshot and all deposits land via
//   atomicAdd, becoming visible only on the NEXT tick. This "all-ants-read-then-
//   write" (Jacobi-style) update order is the natural GPU formulation. Diffusion
//   and evaporation are likewise parallelized one-cell-per-thread. Because each
//   ant carries its own SplitMix64 stream (deterministically derived from --seed),
//   runs are reproducible; the RNG *algorithm* matches the other ports, though the
//   per-ant stream partition differs from the single sequential stream (an
//   unavoidable consequence of parallelism). The emergent signature is unchanged.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define W 56
#define H 28
#define N (W * H)
#define REGION 2
#define DIFFUSE_D 0.03   /* food-trail diffusion rate (Brownian breadth, §3.1/§4.6) */

// 8 neighbours in THIS order (matches the Python DIRS exactly).
__device__ __constant__ int DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
__device__ __constant__ int DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

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

// ---- geometry helpers (device) --------------------------------------------
__device__ static inline bool d_free(const uint8_t *obstacle, int x, int y) {
    return x >= 0 && x < W && y >= 0 && y < H && !obstacle[y * W + x];
}

// nest = (5, H/2), food = (W-6, H/2)
#define NEST_X 5
#define NEST_Y (H / 2)
#define FOOD_X (W - 6)
#define FOOD_Y (H / 2)

__device__ static inline bool d_at_nest(int x, int y) {
    return abs(x - NEST_X) <= REGION && abs(y - NEST_Y) <= REGION;
}
__device__ static inline bool d_at_food(int x, int y) {
    return abs(x - FOOD_X) <= REGION && abs(y - FOOD_Y) <= REGION;
}

// ---- kernels ---------------------------------------------------------------

// 1a. Nest emits +6.0 home pheromone at every FREE cell in the nest region.
__global__ void k_emit_home(double *home, const uint8_t *obstacle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int span = 2 * REGION + 1;               // 5x5 region
    if (idx >= span * span) return;
    int dx = idx % span - REGION;
    int dy = idx / span - REGION;
    int x = NEST_X + dx, y = NEST_Y + dy;
    if (d_free(obstacle, x, y)) home[y * W + x] += 6.0;
}

// 1b. One Jacobi diffusion pass of the home field (simultaneous, uses a copy).
//   new = (self + sum over FREE 8-neighbours) / (1 + count_free_neighbours)
__global__ void k_diffuse_home(const double *home, double *home_new,
                               const uint8_t *obstacle) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N) return;
    int x = cell % W, y = cell / W;
    if (!d_free(obstacle, x, y)) { home_new[cell] = home[cell]; return; }
    double s = home[cell];
    int c = 1;
    for (int k = 0; k < 8; ++k) {
        int xx = x + DX[k], yy = y + DY[k];
        if (d_free(obstacle, xx, yy)) { s += home[yy * W + xx]; c += 1; }
    }
    home_new[cell] = s / (double)c;
}

// 1.5. The food trail SPREADS a little (Brownian breadth, §3.1/§4.6): nearby sub-trails
//      "merge together into a trace." One cell per thread, Jacobi (reads `food`, writes
//      `food_new`). Free neighbours only; new = old + D*(mean_free_nbrs - old).
__global__ void k_diffuse_food(const double *food, double *food_new,
                               const uint8_t *obstacle, double D) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N) return;
    int x = cell % W, y = cell / W;
    if (!d_free(obstacle, x, y)) { food_new[cell] = food[cell]; return; }
    double s = 0.0;
    int c = 0;
    for (int k = 0; k < 8; ++k) {
        int xx = x + DX[k], yy = y + DY[k];
        if (d_free(obstacle, xx, yy)) { s += food[yy * W + xx]; c += 1; }
    }
    double cur = food[cell];
    food_new[cell] = c ? cur + D * (s / (double)c - cur) : cur;
}

// 2. One thread per ant. All ants read the SAME field snapshot; deposits and the
//    delivery counter are resolved by atomicAdd (the stigmergic race made literal).
__global__ void k_step(int *ant_x, int *ant_y, uint8_t *ant_carry,
                       uint64_t *ant_state, const double *food, const double *home,
                       double *food_out, const uint8_t *obstacle,
                       unsigned long long *food_qty, int *deliveries,
                       int n_ants, double deposit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ants) return;

    int x = ant_x[i], y = ant_y[i];
    bool carrying = ant_carry[i];
    uint64_t st = ant_state[i];

    // Rule 2, fully local: follow the field that leads where you're going.
    const double *field = carrying ? home : food;
    double wts[8];
    double tot = 0.0;
    for (int k = 0; k < 8; ++k) {
        int nx = x + DX[k], ny = y + DY[k];
        if (!d_free(obstacle, nx, ny)) { wts[k] = 0.0; continue; }   // rule 1
        wts[k] = 1.0 + field[ny * W + nx] * 6.0;                     // Brownian floor + bias
        tot += wts[k];
    }

    if (tot > 0.0) {                          // else boxed in -> stay put this tick
        double r = sm64_float(&st) * tot;
        double acc = 0.0;
        for (int k = 0; k < 8; ++k) {
            acc += wts[k];
            if (r <= acc) {                  // note the <=, matches the Python
                x += DX[k]; y += DY[k];
                break;
            }
        }
    }

    // rule 3: carriers lay the FOOD trail (atomicAdd resolves same-cell races).
    if (carrying) atomicAdd(&food_out[y * W + x], deposit);

    // rule 4 / rule 5
    if (d_at_food(x, y) && !carrying && *food_qty > 0) {
        carrying = true;
        atomicAdd(food_qty, (unsigned long long)(-1));   // decrement (never exhausts)
    } else if (d_at_nest(x, y) && carrying) {
        carrying = false;
        atomicAdd(deliveries, 1);
    }

    ant_x[i] = x; ant_y[i] = y; ant_carry[i] = carrying; ant_state[i] = st;
}

// 3. Evaporate BOTH fields (the entropy leak), one cell per thread.
__global__ void k_evaporate(double *food, double *home, double keep) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N) return;
    food[cell] *= keep;
    home[cell] *= keep;
}

// ---- host driver -----------------------------------------------------------
static void render_ascii(const double *food, const int *ax, const int *ay, int n_ants) {
    double peak = 0.0;
    for (int i = 0; i < N; ++i) if (food[i] > peak) peak = food[i];
    if (peak <= 0.0) peak = 1.0;
    const char *shades = " .:-=+*#%@";
    int nshades = 10;
    // mark ant positions
    static uint8_t antpos[N];
    memset(antpos, 0, sizeof(antpos));
    for (int i = 0; i < n_ants; ++i) antpos[ay[i] * W + ax[i]] = 1;
    printf("\nGo to the Ant — emergent trail (nest N <-> food F, wall '|', pheromone density):\n\n");
    for (int y = 0; y < H; ++y) {
        char line[W + 1];
        for (int x = 0; x < W; ++x) {
            char c;
            if (x == NEST_X && y == NEST_Y) c = 'N';
            else if (x == FOOD_X && y == FOOD_Y) c = 'F';
            else if (antpos[y * W + x]) c = 'o';
            else {
                int lvl = (int)((food[y * W + x] / peak) * (nshades - 1));
                if (lvl < 0) lvl = 0; if (lvl > nshades - 1) lvl = nshades - 1;
                c = shades[lvl];
            }
            line[x] = c;
        }
        line[W] = '\0';
        printf("%s\n", line);
    }
}

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

int main(int argc, char **argv) {
    int ticks = 3000, n_ants = 90;
    double evap = 0.015, deposit = 1.0;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ants") && i + 1 < argc) n_ants = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--evap") && i + 1 < argc) evap = atof(argv[++i]);
        else if (!strcmp(argv[i], "--deposit") && i + 1 < argc) deposit = atof(argv[++i]);
    }

    // No wall by default -> obstacle grid all-free.
    uint8_t *h_obstacle = (uint8_t *)calloc(N, 1);

    // Per-ant SplitMix64 streams, derived deterministically from --seed so runs
    // reproduce (see OPERATIONALIZED note in the header).
    int   *h_ax = (int *)malloc(n_ants * sizeof(int));
    int   *h_ay = (int *)malloc(n_ants * sizeof(int));
    uint8_t *h_carry = (uint8_t *)calloc(n_ants, 1);
    uint64_t *h_state = (uint64_t *)malloc(n_ants * sizeof(uint64_t));
    uint64_t master = seed;
    for (int i = 0; i < n_ants; ++i) {
        h_ax[i] = NEST_X; h_ay[i] = NEST_Y;
        h_state[i] = sm64_next(&master);      // each ant seeded from the master stream
    }

    // device buffers
    double *d_food, *d_food2, *d_home, *d_home2;
    uint8_t *d_obstacle, *d_carry;
    int *d_ax, *d_ay, *d_deliveries;
    uint64_t *d_state;
    unsigned long long *d_food_qty;
    CK(cudaMalloc(&d_food, N * sizeof(double)));
    CK(cudaMalloc(&d_food2, N * sizeof(double)));
    CK(cudaMalloc(&d_home, N * sizeof(double)));
    CK(cudaMalloc(&d_home2, N * sizeof(double)));
    CK(cudaMalloc(&d_obstacle, N));
    CK(cudaMalloc(&d_carry, n_ants));
    CK(cudaMalloc(&d_ax, n_ants * sizeof(int)));
    CK(cudaMalloc(&d_ay, n_ants * sizeof(int)));
    CK(cudaMalloc(&d_state, n_ants * sizeof(uint64_t)));
    CK(cudaMalloc(&d_deliveries, sizeof(int)));
    CK(cudaMalloc(&d_food_qty, sizeof(unsigned long long)));

    CK(cudaMemset(d_food, 0, N * sizeof(double)));
    CK(cudaMemset(d_home, 0, N * sizeof(double)));
    CK(cudaMemcpy(d_obstacle, h_obstacle, N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_carry, h_carry, n_ants, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ax, h_ax, n_ants * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ay, h_ay, n_ants * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_state, h_state, n_ants * sizeof(uint64_t), cudaMemcpyHostToDevice));
    int zero = 0; unsigned long long fq = 1000000000ULL;
    CK(cudaMemcpy(d_deliveries, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_food_qty, &fq, sizeof(unsigned long long), cudaMemcpyHostToDevice));

    double keep = 1.0 - evap;
    int cellBlocks = (N + 255) / 256;
    int antBlocks  = (n_ants + 127) / 128;
    int regionCells = (2 * REGION + 1) * (2 * REGION + 1);

    // history: 20 evenly-spaced cumulative-delivery samples across the run
    int stepEvery = ticks / 20; if (stepEvery < 1) stepEvery = 1;
    int hist[64]; int nhist = 0;

    for (int t = 0; t < ticks; ++t) {
        // 1. nest broadcasts the home gradient (emit + one Jacobi diffusion pass)
        k_emit_home<<<(regionCells + 31) / 32, 32>>>(d_home, d_obstacle);
        k_diffuse_home<<<cellBlocks, 256>>>(d_home, d_home2, d_obstacle);
        { double *tmp = d_home; d_home = d_home2; d_home2 = tmp; }   // swap in the new field

        // 2. all ants step in parallel. To keep the "all-ants-read-then-write"
        //    (Jacobi) semantics deterministic, they READ the current food field
        //    (d_food) and DEPOSIT into a fresh copy (d_food2); no thread ever reads
        //    a cell another thread is depositing into this tick. Start d_food2 as a
        //    copy of d_food so the persistent field carries over, then deposits
        //    (atomicAdd, the stigmergic race made literal) land on top.
        CK(cudaMemcpy(d_food2, d_food, N * sizeof(double), cudaMemcpyDeviceToDevice));
        k_step<<<antBlocks, 128>>>(d_ax, d_ay, d_carry, d_state, d_food, d_home,
                                   d_food2, d_obstacle, d_food_qty, d_deliveries,
                                   n_ants, deposit);
        { double *tmp = d_food; d_food = d_food2; d_food2 = tmp; }   // new food field

        // 2.5 diffuse the food trail a little (breadth); Jacobi read d_food -> d_food2, swap
        k_diffuse_food<<<cellBlocks, 256>>>(d_food, d_food2, d_obstacle, DIFFUSE_D);
        { double *tmp = d_food; d_food = d_food2; d_food2 = tmp; }   // diffused food field

        // 3. evaporate both fields (the entropy leak)
        k_evaporate<<<cellBlocks, 256>>>(d_food, d_home, keep);

        if (t % stepEvery == 0 && nhist < 64) {
            int d; CK(cudaMemcpy(&d, d_deliveries, sizeof(int), cudaMemcpyDeviceToHost));
            hist[nhist++] = d;
        }
    }
    CK(cudaDeviceSynchronize());

    // pull results back for reporting
    double *h_food = (double *)malloc(N * sizeof(double));
    CK(cudaMemcpy(h_food, d_food, N * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_ax, d_ax, n_ants * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_ay, d_ay, n_ants * sizeof(int), cudaMemcpyDeviceToHost));
    int deliveries; CK(cudaMemcpy(&deliveries, d_deliveries, sizeof(int), cudaMemcpyDeviceToHost));

    render_ascii(h_food, h_ax, h_ay, n_ants);
    printf("\nfood delivered to nest over %d ticks: %d\n", ticks, deliveries);
    printf("deliveries(t):");
    for (int i = 0; i < nhist; ++i) printf(" %d", hist[i]);
    printf("\n");

    // cleanup
    cudaFree(d_food); cudaFree(d_food2); cudaFree(d_home); cudaFree(d_home2); cudaFree(d_obstacle);
    cudaFree(d_carry); cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_state);
    cudaFree(d_deliveries); cudaFree(d_food_qty);
    free(h_obstacle); free(h_ax); free(h_ay); free(h_carry); free(h_state); free(h_food);
    return 0;
}
