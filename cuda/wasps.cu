// wasps.cu — "Go to the Ant" §3.4, Wasp Task Differentiation, CUDA port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
// §3.4, after Theraulaz, Goss, Gervet & Deneubourg (1991).
//
// Faithful port of the authoritative Python reference (wasps.py). Mature Polistes
// wasps — genetically IDENTICAL — split into a single Chief, a band of Foragers,
// and a band of Nurses, with no HR department and nobody computing the proportions.
// Three interacting rules:
//   1. FACE-OFFS. When two wasps meet, j beats i with the Fermi probability
//      p = 1/(1 + e^(h*(F_i - F_j)))   [PAPER §3.4 VERBATIM]. A quantum of Force
//      passes loser -> winner (force conserved).
//   2. BROOD DEMAND.  D(t) = D(t-1) + appetite - W, W = food-work by all foragers.
//   3. FORAGE?  A wasp near the brood forages with Fermi probability
//      p = 1/(1 + e^(hf*(sig_j - D)))  [PAPER §3.4 VERBATIM]. Foraging LOWERS its
//      threshold sig by xi (learning); not foraging RAISES sig by phi (forgetting).
// Force is MOBILITY. The joint (Force, Threshold) distribution self-separates into
// three castes: Chief (1, high force, high threshold), Foragers (few, high force,
// low threshold), Nurses (majority, low force).
//
// OPERATIONALIZED provenance (preserved from the Python source):
//   - ENTROPY LEAK (§4.6): F[k] = max(0, F[k]*(1-leak)+gen) replaces an ad-hoc force
//     CAP. A steady leak+gen bounds the hierarchy naturally (equilibrium mean ~ gen/
//     leak) so no one super-wasp runs away. It shifts the force distribution, which
//     couples into the demand/threshold balance — a real, documented tradeoff.
//   - SPATIALITY PROXY: dominance = (F/Fmax)^4 suppresses only the single top wasp's
//     foraging (the Chief "wanders and faces off", is rarely near the brood), which
//     restores the Chief's HIGH threshold. F~0.7*Fmax foragers are barely touched.
//
// ---------------------------------------------------------------------------
// THE CUDA LENS (this port's distinct view of the system):
//   State lives in GPU global memory (F[], sig[] arrays). The parallel rules —
//   the entropy leak and the foraging/threshold response — run one-thread-per-wasp;
//   the shared brood-work counter W is a literal atomicAdd race across all foragers.
//   But the FACE-OFF spine is irreducibly serial: each face-off reads the freshest
//   F of a RANDOM pair, transfers a force quantum between exactly that pair, and the
//   next face-off must see the result — a dependency chain through one shared RNG
//   stream. CUDA makes the split explicit: the parallel castes emerge in a field of
//   threads, but the dominance ordering that grounds them is forged one duel at a
//   time.
//
// OPERATIONALIZED DEVIATIONS (language-forced, marked and reported):
//   1. FACE-OFFS ARE SERIALIZED. The n//3 face-offs run in a single-thread kernel
//      (<<<1,1>>>) over one RNG stream, exactly matching the Python loop order —
//      because force conservation between a random pair with read-after-write on F
//      is a genuine serial dependency, not parallelizable without changing the
//      dynamics. This preserves the paper's graded hierarchy.
//   2. FORAGING RUNS IN PARALLEL. Each wasp reads the SAME pre-forage snapshot of D
//      and Fmax and updates its own sig; the brood-work W is summed via atomicAdd.
//      Each wasp carries its OWN SplitMix64 stream (deterministically derived from
//      --seed), so the foraging random draws come from per-wasp streams rather than
//      the single sequential Python stream. The RNG *algorithm* is identical to the
//      other ports; the stream partition differs (an unavoidable consequence of
//      parallelism). The emergent 3-caste signature is robust and unchanged.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

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
// randrange(n): next() % n  (cross-port convention)
__host__ __device__ static inline uint64_t sm64_randrange(uint64_t *state, uint64_t n) {
    return sm64_next(state) % n;
}
// uniform(a,b): a + random_float()*(b-a)
__host__ static inline double sm64_uniform(uint64_t *state, double a, double b) {
    return a + sm64_float(state) * (b - a);
}

// ---- model constants (match wasps.py) --------------------------------------
#define SIGMAX 4.0

// ---- kernels ---------------------------------------------------------------

// Rule 1: FACE-OFFS. SERIAL by nature (random pair, force conserved, read-after-
// write on F, one shared RNG stream). One thread performs all n//3 duels in the
// Python loop order. OPERATIONALIZED DEVIATION #1.
__global__ void k_faceoffs(double *F, uint64_t *faceoff_state, int n,
                           double h, double q) {
    uint64_t st = *faceoff_state;
    int rounds = n / 3;
    for (int r = 0; r < rounds; ++r) {
        int i = (int)sm64_randrange(&st, (uint64_t)n);   // Python: i = rng.randrange(n)
        int j = (int)sm64_randrange(&st, (uint64_t)n);   // Python: j = rng.randrange(n)
        if (i == j) continue;
        double pj = 1.0 / (1.0 + exp(h * (F[i] - F[j]))); // PAPER §3.4 VERBATIM
        int w, l;
        if (sm64_float(&st) < pj) { w = j; l = i; } else { w = i; l = j; }
        double t = q < F[l] ? q : F[l];                   // t = min(q, F[loser])
        F[w] += t; F[l] -= t;                             // force conserved
    }
    *faceoff_state = st;
}

// Rule "entropy leak": F[k] = max(0, F[k]*(1-leak)+gen). Fully parallel.
__global__ void k_leak(double *F, int n, double leak, double gen) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double v = F[k] * (1.0 - leak) + gen;
    F[k] = v > 0.0 ? v : 0.0;
}

// Fmax reduction (n is tiny; a single-thread scan keeps it simple and exact).
__global__ void k_fmax(const double *F, int n, double *fmax_out) {
    double m = F[0];
    for (int k = 1; k < n; ++k) if (F[k] > m) m = F[k];
    *fmax_out = m > 0.0 ? m : 1.0;                        // Python: max(F) or 1.0
}

// Rules 2 & 3: FORAGING + demand contribution. One thread per wasp. Reads the same
// pre-forage snapshot of D and Fmax; the shared brood-work W is an atomicAdd race
// across all foragers (the CUDA lens made literal). OPERATIONALIZED DEVIATION #2.
__global__ void k_forage(double *F, double *sig, uint64_t *state, int n,
                         double hf, double xi, double phi, double mob,
                         double D, double Fmax, int *W) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint64_t st = state[k];
    double pf = 1.0 / (1.0 + exp(hf * (sig[k] - D)));    // PAPER §3.4 VERBATIM
    double ratio = F[k] / Fmax;
    double dom = ratio * ratio * ratio * ratio;          // (F/Fmax)^4 spatiality proxy
    if (sm64_float(&st) < pf * (1.0 - dom)) {            // stimulated AND not dominating
        double s = sig[k] - xi;                          // learns: threshold drops
        sig[k] = s > 0.0 ? s : 0.0;
        if (F[k] > mob) atomicAdd(W, 1);                 // mobile enough to hunt -> work
    } else {
        double s = sig[k] + phi;                         // forgets: threshold rises
        sig[k] = s < SIGMAX ? s : SIGMAX;
    }
    state[k] = st;
}

// ---- host: caste classification (matches wasps.py castes()) ----------------
static int argmax_F(const double *F, int n) {
    int c = 0;
    for (int k = 1; k < n; ++k) if (F[k] > F[c]) c = k;
    return c;
}
static double median_sig(const double *sig, int n) {
    double *tmp = (double *)malloc(n * sizeof(double));
    memcpy(tmp, sig, n * sizeof(double));
    // simple insertion sort (n=80)
    for (int i = 1; i < n; ++i) {
        double v = tmp[i]; int j = i - 1;
        while (j >= 0 && tmp[j] > v) { tmp[j + 1] = tmp[j]; --j; }
        tmp[j + 1] = v;
    }
    double m = tmp[n / 2];   // Python: sorted(sig)[n//2]
    free(tmp);
    return m;
}
static double median_F(const double *F, int n) {
    double *tmp = (double *)malloc(n * sizeof(double));
    memcpy(tmp, F, n * sizeof(double));
    for (int i = 1; i < n; ++i) {
        double v = tmp[i]; int j = i - 1;
        while (j >= 0 && tmp[j] > v) { tmp[j + 1] = tmp[j]; --j; }
        tmp[j + 1] = v;
    }
    double m = tmp[n / 2];
    free(tmp);
    return m;
}

// caste of wasp k: 0=Chief, 1=Forager, 2=Nurse
static int caste_of(int k, int chief, const double *F, const double *sig,
                    double smed, double mob) {
    if (k == chief) return 0;
    if (F[k] > mob && sig[k] <= smed) return 1;
    return 2;
}

// ---- host: ASCII (F,sigma) landscape (matches wasps.py landscape()) --------
static void landscape(const double *F, const double *sig, int n) {
    int cols = 48, rows = 16;
    double fmn = F[0], fmx = F[0], smn = sig[0], smx = sig[0];
    for (int k = 1; k < n; ++k) {
        if (F[k] < fmn) fmn = F[k];
        if (F[k] > fmx) fmx = F[k];
        if (sig[k] < smn) smn = sig[k];
        if (sig[k] > smx) smx = sig[k];
    }
    char *grid = (char *)malloc(rows * cols);
    memset(grid, ' ', rows * cols);
    int chief = argmax_F(F, n);
    double fmed = median_F(F, n);
    for (int k = 0; k < n; ++k) {
        int x = (int)((F[k] - fmn) / (fmx - fmn + 1e-9) * (cols - 1));
        int y = (int)((sig[k] - smn) / (smx - smn + 1e-9) * (rows - 1));
        char mark = (k == chief) ? 'C' : (F[k] >= fmed ? 'F' : 'n');
        grid[(rows - 1 - y) * cols + x] = mark;
    }
    printf("\n  (F,sigma) landscape - x = Force ->, y = Threshold ^ | C chief, F forager, n nurse:\n\n");
    for (int r = 0; r < rows; ++r) {
        printf("   ");
        for (int cc = 0; cc < cols; ++cc) putchar(grid[r * cols + cc]);
        putchar('\n');
    }
    free(grid);
}

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

int main(int argc, char **argv) {
    int ticks = 4000, n = 80;
    uint64_t seed = 0;
    // model parameters (match wasps.py defaults)
    double h = 1.1, hf = 3.0, quantum = 0.10;
    double xi = 0.02, phi = 0.012, mob = 1.6, leak = 0.004, gen = 0.005;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--wasps") || !strcmp(argv[i], "--ants")) && i + 1 < argc)
            n = atoi(argv[++i]);
    }
    double appetite = 0.075 * (double)n;

    // ---- host init (match wasps.py __init__ RNG order: all F then all sig) ----
    double *h_F = (double *)malloc(n * sizeof(double));
    double *h_sig = (double *)malloc(n * sizeof(double));
    uint64_t *h_state = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t init_stream = seed;                       // one stream for the tiny spread
    for (int k = 0; k < n; ++k) h_F[k] = 1.0 + sm64_uniform(&init_stream, -0.05, 0.05);
    for (int k = 0; k < n; ++k) h_sig[k] = 1.6 + sm64_uniform(&init_stream, -0.05, 0.05);
    // Derive the serial face-off stream and the per-wasp foraging streams from a
    // second master (deterministic in --seed). CUDA's stream partition, marked above.
    uint64_t master = seed + 0x1234567ULL;
    uint64_t h_faceoff = sm64_next(&master);
    for (int k = 0; k < n; ++k) h_state[k] = sm64_next(&master);

    double D = 2.0;

    // ---- device buffers ----
    double *d_F, *d_sig, *d_fmax;
    uint64_t *d_state, *d_faceoff;
    int *d_W;
    CK(cudaMalloc(&d_F, n * sizeof(double)));
    CK(cudaMalloc(&d_sig, n * sizeof(double)));
    CK(cudaMalloc(&d_fmax, sizeof(double)));
    CK(cudaMalloc(&d_state, n * sizeof(uint64_t)));
    CK(cudaMalloc(&d_faceoff, sizeof(uint64_t)));
    CK(cudaMalloc(&d_W, sizeof(int)));
    CK(cudaMemcpy(d_F, h_F, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sig, h_sig, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_state, h_state, n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_faceoff, &h_faceoff, sizeof(uint64_t), cudaMemcpyHostToDevice));

    int blocks = (n + 127) / 128;

    // history: Forager/Nurse split sampled every ticks//12 (matches Python)
    int sampleEvery = ticks / 12; if (sampleEvery < 1) sampleEvery = 1;
    int histF[64], histN[64], nhist = 0;

    for (int t = 0; t < ticks; ++t) {
        // (1) face-offs — SERIAL single-thread kernel (OPERATIONALIZED #1)
        k_faceoffs<<<1, 1>>>(d_F, d_faceoff, n, h, quantum);
        // (2) entropy leak — parallel
        k_leak<<<blocks, 128>>>(d_F, n, leak, gen);
        // (3) foraging + demand — parallel, W via atomicAdd (OPERATIONALIZED #2)
        k_fmax<<<1, 1>>>(d_F, n, d_fmax);
        double h_fmax; CK(cudaMemcpy(&h_fmax, d_fmax, sizeof(double), cudaMemcpyDeviceToHost));
        int zeroW = 0; CK(cudaMemcpy(d_W, &zeroW, sizeof(int), cudaMemcpyHostToDevice));
        k_forage<<<blocks, 128>>>(d_F, d_sig, d_state, n, hf, xi, phi, mob, D, h_fmax, d_W);
        int W; CK(cudaMemcpy(&W, d_W, sizeof(int), cudaMemcpyDeviceToHost));
        D = D + appetite - (double)W;
        if (D < 0.0) D = 0.0;                            // D = max(0, D + appetite - W)

        if (t % sampleEvery == 0 && nhist < 64) {
            CK(cudaMemcpy(h_F, d_F, n * sizeof(double), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_sig, d_sig, n * sizeof(double), cudaMemcpyDeviceToHost));
            int chief = argmax_F(h_F, n);
            double smed = median_sig(h_sig, n);
            int f = 0, ns = 0;
            for (int k = 0; k < n; ++k) {
                int c = caste_of(k, chief, h_F, h_sig, smed, mob);
                if (c == 1) ++f; else if (c == 2) ++ns;
            }
            histF[nhist] = f; histN[nhist] = ns; ++nhist;
        }
    }
    CK(cudaDeviceSynchronize());

    // ---- pull final state and report ----
    CK(cudaMemcpy(h_F, d_F, n * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_sig, d_sig, n * sizeof(double), cudaMemcpyDeviceToHost));

    int chief = argmax_F(h_F, n);
    double smed = median_sig(h_sig, n);
    // per-caste aggregates
    const char *names[3] = {"Chief", "Forager", "Nurse"};
    int cnt[3] = {0, 0, 0};
    double sumF[3] = {0, 0, 0}, sumS[3] = {0, 0, 0};
    for (int k = 0; k < n; ++k) {
        int c = caste_of(k, chief, h_F, h_sig, smed, mob);
        cnt[c]++; sumF[c] += h_F[k]; sumS[c] += h_sig[k];
    }
    double popMean = 0.0; for (int k = 0; k < n; ++k) popMean += h_F[k]; popMean /= n;

    printf("Emergent castes from %d genetically identical wasps (%d ticks):\n\n", n, ticks);
    for (int c = 0; c < 3; ++c) {
        if (cnt[c] == 0) continue;
        printf("  %-8s n=%3d   mean Force %5.2f   mean Threshold %5.2f\n",
               names[c], cnt[c], sumF[c] / cnt[c], sumS[c] / cnt[c]);
    }
    printf("\n  Chief force %.2f (pop mean %.2f), threshold %.2f\n",
           h_F[chief], popMean, h_sig[chief]);
    printf("  Forager/Nurse split(t):");
    for (int i = 0; i < nhist; ++i) printf(" %d/%d", histF[i], histN[i]);
    printf("\n");
    landscape(h_F, h_sig, n);

    cudaFree(d_F); cudaFree(d_sig); cudaFree(d_fmax);
    cudaFree(d_state); cudaFree(d_faceoff); cudaFree(d_W);
    free(h_F); free(h_sig); free(h_state);
    return 0;
}
