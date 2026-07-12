// wolves.cu — "Go to the Ant" §3.6, Wolves: Surrounding Prey, CUDA port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
// §3.6, after Korf (1992).
//
// Faithful port of the authoritative Python reference (wolves.py). One wolf can't
// kill a moose; the pack must SURROUND it, with no radios and no negotiated
// strategy. Parunak gives two LOCAL rules:
//   1. MOOSE: move to the candidate FARTHEST from its nearest wolf (maximise the
//      min distance to any wolf).
//   2. WOLVES: each wolf moves to minimise  S = d(moose) - k*d(nearest other wolf)
//      [PAPER §3.6 VERBATIM] — get CLOSE to the prey while staying FAR from packmates.
// Attraction (to prey) balanced against repulsion (between wolves) makes the pack
// encircle and PIN the moose; no communication required.
//
// PROVENANCE: the score S = d(moose) - k*d(wolf) is the paper's (Korf 1992).
//   OPERATIONALIZED (preserved from the Python source): a continuous plane and a
//   24-candidate direction search replace the paper's hex grid; k=1.12; the speeds
//   vm=0.6, vw=1.0. No per-step randomness — the run is deterministic given the
//   initial (uniform-random) wolf placement.
//
// ---------------------------------------------------------------------------
// THE CUDA LENS (this port's distinct view of the system):
//   The whole world state — moose (mx,my) and the 6 wolves — lives in GPU global
//   memory. Each agent's decision is an independent argmax/argmin over its 25
//   candidate moves (24 directions + stay), so the natural GPU shape is ONE THREAD
//   PER AGENT, each scanning its candidate fan in registers. The moose runs first
//   (a single-thread argmax kernel), then the 6 wolves run as 6 parallel threads,
//   each reading the just-updated moose and the SHARED pre-step wolf snapshot. There
//   is no shared field and no write contention, so no atomics are needed: the wolves
//   are a pure data-parallel map over the pack.
//
// OPERATIONALIZED DEVIATIONS (language-forced, marked and reported):
//   1. TINY POPULATIONS. There are only 6 wolves and a single moose, and each scans
//      just 25 candidates. Launching a 6-thread wolf kernel and a 1-thread moose
//      kernel is far below any GPU efficiency threshold — the parallelism is
//      NOTIONAL, chosen to express the one-thread-per-agent structure faithfully,
//      not for speed. Marked per the LENS.
//   2. UPDATE ORDER (bit-faithful, not a divergence). The Python already uses a
//      Jacobi-style wolf update: every wolf reads the OLD wolf array W and the NEW
//      moose position, then all new positions are committed together. The CUDA
//      kernels reproduce that order exactly (moose kernel -> wolf kernel reading the
//      old wolf buffer, writing a fresh one, then swap). Each agent scans its 25
//      candidates in the SAME index order as Python with the SAME strict-inequality
//      tie-break, and there is no per-step RNG, so this port is deterministic and
//      bit-comparable with the sequential ports. The init RNG (SplitMix64 from
//      --seed) is consumed in the same order: for each wolf, uniform(0,w) then
//      uniform(0,h).
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#define NCAND 25   // 24 directions + stay
static const double TAU = 6.283185307179586;

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
__host__ __device__ static inline double sm64_uniform(uint64_t *state, double a, double b) {
    return a + sm64_float(state) * (b - a);
}

// ---- candidate generator (matches Python _cands order exactly) ------------
// candidate i in 0..23 = (x + sp*cos(i*TAU/24), y + sp*sin(i*TAU/24));
// candidate 24 = (x, y) (stay). Returns the i-th candidate.
__host__ __device__ static inline void candidate(double x, double y, double sp,
                                                 int i, double *cx, double *cy) {
    if (i < 24) {
        double a = (double)i * TAU / 24.0;
        *cx = x + sp * cos(a);
        *cy = y + sp * sin(a);
    } else {
        *cx = x; *cy = y;
    }
}

__host__ __device__ static inline double hyp(double dx, double dy) {
    return sqrt(dx * dx + dy * dy);
}

// ---- kernels ---------------------------------------------------------------

// Rule 1: the moose flees to the in-bounds candidate FARTHEST from its nearest
// wolf (maximise min-distance). Single-thread argmax — one moose.
__global__ void k_moose(double *mx, double *my, const double *wx, const double *wy,
                        int n_wolves, double vm, int w, int h) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    double ox = *mx, oy = *my;
    double bestx = ox, besty = oy, bd = -1.0;
    for (int i = 0; i < NCAND; ++i) {
        double cx, cy; candidate(ox, oy, vm, i, &cx, &cy);
        if (!(cx >= 0.0 && cx < (double)w && cy >= 0.0 && cy < (double)h)) continue;
        double d = 1e300;
        for (int j = 0; j < n_wolves; ++j) {
            double dd = hyp(cx - wx[j], cy - wy[j]);
            if (dd < d) d = dd;
        }
        if (d > bd) { bd = d; bestx = cx; besty = cy; }   // strict > : first-wins tie-break
    }
    *mx = bestx; *my = besty;
}

// Rule 2: each wolf (one thread) minimises S = d(moose) - k*d(nearest OTHER wolf).
// Reads the NEW moose and the OLD wolf snapshot (wx,wy); writes the fresh buffer.
__global__ void k_wolves(const double *wx, const double *wy, double *nwx, double *nwy,
                         double mx, double my, int n_wolves, double vw, double k,
                         int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_wolves) return;
    double ox = wx[idx], oy = wy[idx];
    double bestx = ox, besty = oy, bs = 1e9;
    for (int i = 0; i < NCAND; ++i) {
        double cx, cy; candidate(ox, oy, vw, i, &cx, &cy);
        if (!(cx >= 0.0 && cx < (double)w && cy >= 0.0 && cy < (double)h)) continue;
        double dm = hyp(cx - mx, cy - my);
        double dother = 1e300; bool any = false;
        for (int j = 0; j < n_wolves; ++j) {
            if (j == idx) continue;
            double dd = hyp(cx - wx[j], cy - wy[j]);
            if (dd < dother) dother = dd;
            any = true;
        }
        if (!any) dother = 0.0;                            // default 0.0 (matches Python)
        double s = dm - k * dother;                        // S = d(moose) - k*d(wolf)
        if (s < bs) { bs = s; bestx = cx; besty = cy; }    // strict < : first-wins tie-break
    }
    nwx[idx] = bestx; nwy[idx] = besty;
}

// ---- host helpers ----------------------------------------------------------

// Largest angular gap (deg) between adjacent wolves as seen from the moose.
// 360/N when evenly ringed -> surrounded; near 360 when all on one side.
static double gap_deg(const double *wx, const double *wy, int n,
                      double mx, double my) {
    if (n < 2) return 360.0;
    double a[64];
    for (int i = 0; i < n; ++i) a[i] = atan2(wy[i] - my, wx[i] - mx);
    // insertion sort ascending
    for (int i = 1; i < n; ++i) {
        double v = a[i]; int j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; --j; }
        a[j + 1] = v;
    }
    double best = 0.0;
    for (int i = 0; i < n; ++i) {
        double g = a[(i + 1) % n] - a[i];
        g = fmod(g, TAU); if (g < 0) g += TAU;             // (..) % TAU, positive
        if (g > best) best = g;
    }
    return best * 180.0 / M_PI;
}

static void render(const double *wx, const double *wy, int n, double mx, double my,
                   int w, int h) {
    char *grid = (char *)malloc(w * h);
    memset(grid, ' ', w * h);
    for (int i = 0; i < n; ++i) {
        int x = ((int)wx[i] % w + w) % w, y = ((int)wy[i] % h + h) % h;
        grid[y * w + x] = 'W';
    }
    int mxi = ((int)mx % w + w) % w, myi = ((int)my % h + h) % h;
    grid[myi * w + mxi] = 'M';
    printf("\nThe hunt (M moose, W wolves - watch the ring close):\n\n");
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) putchar(grid[y * w + x]);
        putchar('\n');
    }
    free(grid);
}

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

int main(int argc, char **argv) {
    int ticks = 260, n_wolves = 6, w = 80, h = 44;
    double vm = 0.6, vw = 1.0, k = 1.12;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--wolves") || !strcmp(argv[i], "--ants")) && i + 1 < argc)
            n_wolves = atoi(argv[++i]);
    }

    // ---- host init: moose at centre; wolves uniform-random ------------------
    // RNG order matches wolves.py exactly: per wolf, uniform(0,w) then uniform(0,h).
    double *h_wx = (double *)malloc(n_wolves * sizeof(double));
    double *h_wy = (double *)malloc(n_wolves * sizeof(double));
    uint64_t rng = seed;
    for (int i = 0; i < n_wolves; ++i) {
        h_wx[i] = sm64_uniform(&rng, 0.0, (double)w);
        h_wy[i] = sm64_uniform(&rng, 0.0, (double)h);
    }
    double h_mx = w / 2.0, h_my = h / 2.0;

    // ---- device buffers -----------------------------------------------------
    double *d_wx, *d_wy, *d_nwx, *d_nwy, *d_mx, *d_my;
    CK(cudaMalloc(&d_wx, n_wolves * sizeof(double)));
    CK(cudaMalloc(&d_wy, n_wolves * sizeof(double)));
    CK(cudaMalloc(&d_nwx, n_wolves * sizeof(double)));
    CK(cudaMalloc(&d_nwy, n_wolves * sizeof(double)));
    CK(cudaMalloc(&d_mx, sizeof(double)));
    CK(cudaMalloc(&d_my, sizeof(double)));
    CK(cudaMemcpy(d_wx, h_wx, n_wolves * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_wy, h_wy, n_wolves * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_mx, &h_mx, sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_my, &h_my, sizeof(double), cudaMemcpyHostToDevice));

    int wolfBlocks = (n_wolves + 31) / 32;

    // history: initial gap, then the post-step gap every ticks//12 ticks (matches Python)
    int sampleEvery = ticks / 12; if (sampleEvery < 1) sampleEvery = 1;
    double hist[64]; int nhist = 0;
    hist[nhist++] = gap_deg(h_wx, h_wy, n_wolves, h_mx, h_my);

    for (int t = 0; t < ticks; ++t) {
        // 1. moose flees (single-thread argmax), reading the OLD wolf snapshot
        k_moose<<<1, 32>>>(d_mx, d_my, d_wx, d_wy, n_wolves, vm, w, h);
        // 2. wolves close in (one thread each), reading the NEW moose + OLD wolves
        //    into a fresh buffer, then swap (Jacobi update, exactly as in Python)
        CK(cudaMemcpy(&h_mx, d_mx, sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&h_my, d_my, sizeof(double), cudaMemcpyDeviceToHost));
        k_wolves<<<wolfBlocks, 32>>>(d_wx, d_wy, d_nwx, d_nwy, h_mx, h_my,
                                     n_wolves, vw, k, w, h);
        { double *t1 = d_wx; d_wx = d_nwx; d_nwx = t1; }
        { double *t2 = d_wy; d_wy = d_nwy; d_nwy = t2; }

        if (t % sampleEvery == 0 && nhist < 64) {
            CK(cudaMemcpy(h_wx, d_wx, n_wolves * sizeof(double), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_wy, d_wy, n_wolves * sizeof(double), cudaMemcpyDeviceToHost));
            hist[nhist++] = gap_deg(h_wx, h_wy, n_wolves, h_mx, h_my);
        }
    }
    CK(cudaDeviceSynchronize());

    // ---- pull final state and report ---------------------------------------
    CK(cudaMemcpy(h_wx, d_wx, n_wolves * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_wy, d_wy, n_wolves * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&h_mx, d_mx, sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&h_my, d_my, sizeof(double), cudaMemcpyDeviceToHost));

    render(h_wx, h_wy, n_wolves, h_mx, h_my, w, h);

    double md = 1e300;
    for (int i = 0; i < n_wolves; ++i) {
        double dd = hyp(h_mx - h_wx[i], h_my - h_wy[i]);
        if (dd < md) md = dd;
    }
    double fg = gap_deg(h_wx, h_wy, n_wolves, h_mx, h_my);
    printf("\nlargest escape gap around the moose: %.0f deg  "
           "(evenly surrounded ~ %d deg) | nearest wolf %.1f\n",
           fg, 360 / n_wolves, md);
    printf("gap deg(t):");
    for (int i = 0; i < nhist; ++i) printf(" %.0f", hist[i]);
    printf("\n");

    cudaFree(d_wx); cudaFree(d_wy); cudaFree(d_nwx); cudaFree(d_nwy);
    cudaFree(d_mx); cudaFree(d_my);
    free(h_wx); free(h_wy);
    return 0;
}
