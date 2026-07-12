// termites.cu — "Go to the Ant" §3.3 Termite Nest Building (Kugler et al. 1990),
// CUDA port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997), §3.3.
//
// Faithful port of the authoritative Python reference (termites.py). Parunak's
// three LOCAL termite rules (§3.3), preserved in spirit:
//   1. Metabolize bodily waste, which contains pheromone. The waste IS the
//      building material (load = min(maxload, load + metab)).
//   2. Wander randomly over the 8 toroidal neighbours, biased toward the
//      strongest local scent (weight = 1 + scent*3).
//   3. Stochastically deposit the carried load. p(deposit) rises with the LOCAL
//      scent density AND the amount carried; a full termite always drops.
// TWO fields:
//   mass  : persistent structural mass (what you see; columns form here).
//   scent : decaying pheromone (biases wandering; evaporates *= (1-decay)/tick).
// Because scent DECAYS, the freshest deposits (the centre of a growing pile)
// smell strongest, so dabs self-concentrate upward into COLUMNS rather than
// spreading. No termite plans the mound.
//
// ---------------------------------------------------------------------------
// THE CUDA LENS (this port's distinct view of the system):
//   Both fields ARE GPU global memory. Termites ARE threads (one per agent).
//   Deposits ARE atomicAdd race-resolution — several termites reinforcing the
//   same growing column in the same instant is exactly the stigmergic
//   superposition the paper describes, made literal by the hardware. Column
//   detection (toroidal local maxima) is itself a parallel-friendly reduction,
//   though here it runs on the host at report time.
//
// OPERATIONALIZED DEVIATION (language-forced, marked and reported):
//   The Python steps termites SEQUENTIALLY: termite k sees the fresh deposits of
//   termites 0..k-1 laid earlier in the same tick. Here termites run in PARALLEL
//   threads, so every termite reads the SAME pre-tick scent snapshot and all
//   deposits land via atomicAdd, becoming visible only on the NEXT tick. This
//   "all-termites-read-then-write" (Jacobi-style) order is the natural GPU
//   formulation. With only a one-tick delay against a field that persists across
//   40000 ticks (scent decays at just 0.02/tick), the positive-feedback loop that
//   grows columns is unchanged — the emergent signature is identical in kind.
//   Each termite carries its own SplitMix64 stream (deterministically derived
//   from --seed); the RNG *algorithm* matches the other ports, but the per-termite
//   stream partition and the parallel deposit order differ from the single
//   sequential Python stream, so exact metric numbers are NOT bit-comparable
//   (an unavoidable consequence of parallelism).
//
// OPERATIONALIZED (from the Python, preserved): the deposit-probability formula
//   p = min(1, 0.01 + 0.55*(load/maxload) + 0.20*local_scent)
//   — the paper §3.3 gives NO formula, only "probability rises with local density
//   AND load". This concrete form is a modelling choice, tagged as such.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define W 58
#define H 34
#define N (W * H)

// 8 neighbours in THIS order (matches the Python DIRS exactly):
// (-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)
__device__ __constant__ int DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
__device__ __constant__ int DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const int H_DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
static const int H_DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

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
    return sm64_next(state) % n;
}

// ---- kernels ---------------------------------------------------------------

// One thread per termite. All termites read the SAME pre-tick scent snapshot
// (`scent`); deposits land via atomicAdd into `mass` and into a fresh scent copy
// (`scent_out`) — the stigmergic race made literal.
__global__ void k_step(int *tx, int *ty, double *load, uint64_t *tstate,
                        const double *scent, double *scent_out, double *mass,
                        int n, double metab, double maxload) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int x = tx[i], y = ty[i];
    double ld = load[i];
    uint64_t st = tstate[i];

    // rule 1: metabolize -> waste (load) accumulates, capped at maxload.
    ld = ld + metab;
    if (ld > maxload) ld = maxload;

    // rule 2: wander over the 8 toroidal neighbours, biased toward strong scent.
    double wts[8];
    double tot = 0.0;
    for (int k = 0; k < 8; ++k) {
        int nx = (x + DX[k] + W) % W;
        int ny = (y + DY[k] + H) % H;
        double w = 1.0 + scent[ny * W + nx] * 3.0;
        wts[k] = w;
        tot += w;
    }
    double r = sm64_float(&st) * tot;
    for (int k = 0; k < 8; ++k) {
        r -= wts[k];
        if (r <= 0.0) {                     // note the <=, matches the Python
            x = (x + DX[k] + W) % W;
            y = (y + DY[k] + H) % H;
            break;
        }
    }

    // rule 3: stochastic deposit — rises with local scent AND load; full drops.
    // OPERATIONALIZED formula (paper gives none); see header.
    double local = scent[y * W + x];
    double p = 0.01 + 0.55 * (ld / maxload) + 0.20 * local;
    if (p > 1.0) p = 1.0;
    if (ld >= maxload || sm64_float(&st) < p) {
        atomicAdd(&mass[y * W + x], ld);
        atomicAdd(&scent_out[y * W + x], ld);
        ld = 0.0;
    }

    tx[i] = x; ty[i] = y; load[i] = ld; tstate[i] = st;
}

// Evaporate the scent field (the entropy leak), one cell per thread.
__global__ void k_evaporate(double *scent, double keep) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N) return;
    scent[cell] *= keep;
}

// ---- host helpers ----------------------------------------------------------

// columns() = count toroidal local maxima with mass above 0.15*peak.
static int columns(const double *mass, double *peak_out) {
    double peak = 0.0;
    for (int i = 0; i < N; ++i) if (mass[i] > peak) peak = mass[i];
    if (peak_out) *peak_out = peak;
    if (peak <= 0.0) return 0;
    double cut = peak * 0.15;
    int cnt = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double v = mass[y * W + x];
            if (v < cut) continue;
            bool isMax = true;
            for (int k = 0; k < 8; ++k) {
                int nx = (x + H_DX[k] + W) % W;
                int ny = (y + H_DY[k] + H) % H;
                if (v < mass[ny * W + nx]) { isMax = false; break; }
            }
            if (isMax) cnt += 1;
        }
    }
    return cnt;
}

static void render_ascii(const double *mass) {
    double peak = 0.0;
    for (int i = 0; i < N; ++i) if (mass[i] > peak) peak = mass[i];
    if (peak <= 0.0) peak = 1.0;
    const char *shades = " .:-=+*#%@";
    int nshades = 10;
    printf("\nTermite mound (top-down mass density — columns emerge as bright cores):\n\n");
    for (int y = 0; y < H; ++y) {
        char line[W + 1];
        for (int x = 0; x < W; ++x) {
            int lvl = (int)((mass[y * W + x] / peak) * (nshades - 1));
            if (lvl < 0) lvl = 0;
            if (lvl > nshades - 1) lvl = nshades - 1;
            line[x] = shades[lvl];
        }
        line[W] = '\0';
        printf("%s\n", line);
    }
}

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

int main(int argc, char **argv) {
    int ticks = 40000, n = 70;
    double decay = 0.02, metab = 0.4, maxload = 6.0;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--termites") || !strcmp(argv[i], "--ants")) && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--decay") && i + 1 < argc) decay = atof(argv[++i]);
    }

    // Per-termite SplitMix64 streams, derived deterministically from --seed so
    // runs reproduce (see OPERATIONALIZED note in the header). Each termite also
    // draws its initial toroidal position from its own stream.
    int      *h_tx    = (int *)malloc(n * sizeof(int));
    int      *h_ty    = (int *)malloc(n * sizeof(int));
    double   *h_load  = (double *)calloc(n, sizeof(double));
    uint64_t *h_state = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t master = seed;
    for (int i = 0; i < n; ++i) {
        uint64_t st = sm64_next(&master);
        h_tx[i] = (int)sm64_randrange(&st, (uint64_t)W);
        h_ty[i] = (int)sm64_randrange(&st, (uint64_t)H);
        h_state[i] = st;
    }

    // device buffers
    double *d_mass, *d_scent, *d_scent2, *d_load;
    int *d_tx, *d_ty;
    uint64_t *d_state;
    CK(cudaMalloc(&d_mass, N * sizeof(double)));
    CK(cudaMalloc(&d_scent, N * sizeof(double)));
    CK(cudaMalloc(&d_scent2, N * sizeof(double)));
    CK(cudaMalloc(&d_load, n * sizeof(double)));
    CK(cudaMalloc(&d_tx, n * sizeof(int)));
    CK(cudaMalloc(&d_ty, n * sizeof(int)));
    CK(cudaMalloc(&d_state, n * sizeof(uint64_t)));

    CK(cudaMemset(d_mass, 0, N * sizeof(double)));
    CK(cudaMemset(d_scent, 0, N * sizeof(double)));
    CK(cudaMemcpy(d_load, h_load, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tx, h_tx, n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ty, h_ty, n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_state, h_state, n * sizeof(uint64_t), cudaMemcpyHostToDevice));

    double keep = 1.0 - decay;
    int cellBlocks = (N + 255) / 256;
    int antBlocks = (n + 127) / 128;

    double *h_mass = (double *)malloc(N * sizeof(double));

    // history: columns(t) sampled like the Python (every ticks/12 ticks).
    int stepEvery = ticks / 12; if (stepEvery < 1) stepEvery = 1;
    int hist[64]; int nhist = 0;

    for (int t = 0; t < ticks; ++t) {
        // Read-then-write (Jacobi) scent update: start scent_out as a copy of the
        // current scent so persistent pheromone carries over, then all termites
        // read the OLD snapshot and deposit into the copy via atomicAdd.
        CK(cudaMemcpy(d_scent2, d_scent, N * sizeof(double), cudaMemcpyDeviceToDevice));
        k_step<<<antBlocks, 128>>>(d_tx, d_ty, d_load, d_state,
                                   d_scent, d_scent2, d_mass, n, metab, maxload);
        { double *tmp = d_scent; d_scent = d_scent2; d_scent2 = tmp; }  // new scent field

        // evaporate the scent (the entropy leak)
        k_evaporate<<<cellBlocks, 256>>>(d_scent, keep);

        if (t % stepEvery == 0 && nhist < 64) {
            CK(cudaMemcpy(h_mass, d_mass, N * sizeof(double), cudaMemcpyDeviceToHost));
            hist[nhist++] = columns(h_mass, 0);
        }
    }
    CK(cudaDeviceSynchronize());

    // pull the mound back for reporting
    CK(cudaMemcpy(h_mass, d_mass, N * sizeof(double), cudaMemcpyDeviceToHost));

    render_ascii(h_mass);
    double peak = 0.0;
    int cnt = columns(h_mass, &peak);
    printf("\ndistinct columns (local maxima): %d | tallest column mass: %.0f\n", cnt, peak);
    printf("columns(t):");
    for (int i = 0; i < nhist; ++i) printf(" %d", hist[i]);
    printf("\n");

    cudaFree(d_mass); cudaFree(d_scent); cudaFree(d_scent2);
    cudaFree(d_load); cudaFree(d_tx); cudaFree(d_ty); cudaFree(d_state);
    free(h_tx); free(h_ty); free(h_load); free(h_state); free(h_mass);
    return 0;
}
