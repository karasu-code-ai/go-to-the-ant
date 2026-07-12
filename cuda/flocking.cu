// flocking.cu — "Go to the Ant" §3.5 Birds & Fish: Flocking, CUDA port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
// §3.5 (Reynolds 1987 "boids"; Heppner 1990).
//
// Faithful port of the authoritative Python reference (flocking.py). Reynolds'
// three LOCAL rules, each a steering vector from the neighbours inside a
// perception radius, are preserved:
//   1. SEPARATION — keep a minimum distance from the nearest birds.
//   2. ALIGNMENT  — match velocity (speed + heading) to nearby birds.
//   3. COHESION   — steer toward the centre of the local flock.
// A single coherent, banking flock EMERGES from these three urges with no leader.
//
// PROVENANCE: the three rules are the paper's (Reynolds' boids). The perception
// radius, the separation distance, and the three weights are OPERATIONALIZED —
// Parunak lists the rules but gives no numbers (Reynolds 1987 is the primary
// source for tuned constants). These provenance tags mirror the Python source.
//
// ---------------------------------------------------------------------------
// THE CUDA LENS (this port's distinct view of the system):
//   Every bird's state (px,py,vx,vy) lives in GPU global memory; one thread per
//   bird. Unlike the foraging port, flocking needs NO atomicAdd: the Python
//   reference is already a Jacobi update — it computes ALL new velocities from
//   the pre-tick snapshot (into nvx/nvy copies) and only then moves every bird.
//   That "all-birds-read-old, all-birds-write-new" structure IS the natural GPU
//   formulation, so the CUDA port is a data-parallel MIRROR of the sequential
//   reference with no race to resolve. n=90 birds -> n threads, each doing the
//   O(n) neighbour scan; the whole flock's step is one kernel launch. The
//   emergent order parameter is what the hardware makes visible in parallel.
//
// OPERATIONALIZED DEVIATION (language-forced, marked and reported):
//   The random INIT (uniform positions + headings) is done SEQUENTIALLY on the
//   host so the SplitMix64 stream is consumed in EXACTLY the Python order
//   (all px, then all py, then all headings) — this keeps the init bit-identical
//   to the sequential ports. The per-tick dynamics carry NO randomness at all
//   (fully deterministic given the init), and each thread scans neighbours j in
//   ascending index order, so the neighbour-sum order matches the reference too.
//   No serialized face-off, no argmax, no shared RNG stream exists in this system.
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
__host__ static inline double sm64_uniform(uint64_t *state, double a, double b) {
    return a + sm64_float(state) * (b - a);
}

// ---- flocking step: one thread per bird -----------------------------------
// Reads the pre-tick snapshot (px,py,vx,vy) and writes the new velocity
// (nvx,nvy). No cross-thread writes => no atomics needed (see CUDA LENS note).
__global__ void k_step(const double *px, const double *py,
                       const double *vx, const double *vy,
                       double *nvx, double *nvy, int n,
                       double w, double h, double perc, double sep_r,
                       double wsep, double wali, double wcoh,
                       double vmax, double turn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double p2 = perc * perc, s2 = sep_r * sep_r;
    double sx = 0, sy = 0, ax = 0, ay = 0, cx = 0, cy = 0;
    int cnt = 0;

    // neighbour scan in ascending index order (matches the Python j-loop, so the
    // floating-point neighbour-sum order is identical to the sequential ports).
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        double dx = px[j] - px[i], dy = py[j] - py[i];
        dx -= w * rint(dx / w);              // toroidal delta (round half-to-even, as Python)
        dy -= h * rint(dy / h);
        double d2 = dx * dx + dy * dy;
        if (d2 > p2) continue;
        cnt += 1;
        ax += vx[j]; ay += vy[j];            // rule 2: alignment (avg neighbour velocity)
        cx += dx;    cy += dy;               // rule 3: cohesion (toward neighbour centre)
        if (d2 < s2 && d2 > 1e-9) {          // rule 1: separation (push from close ones)
            sx -= dx / d2; sy -= dy / d2;
        }
    }

    double outx = vx[i], outy = vy[i];       // default: unchanged if no neighbours
    if (cnt) {
        ax /= cnt; ay /= cnt; cx /= cnt; cy /= cnt;
        // NORMALIZE each urge to a unit vector so the three weights are actually
        // comparable (else the position-scale cohesion swamps velocity-scale align).
        double m;
        double sux = 0, suy = 0, aux = 0, auy = 0, cux = 0, cuy = 0;
        m = hypot(sx, sy);                 if (m > 1e-9) { sux = sx / m; suy = sy / m; }
        double avx = ax - vx[i], avy = ay - vy[i];
        m = hypot(avx, avy);               if (m > 1e-9) { aux = avx / m; auy = avy / m; }
        m = hypot(cx, cy);                 if (m > 1e-9) { cux = cx / m; cuy = cy / m; }
        double accx = wsep * sux + wali * aux + wcoh * cux;
        double accy = wsep * suy + wali * auy + wcoh * cuy;
        double tx = vx[i] + turn * accx, ty = vy[i] + turn * accy;
        double sp = hypot(tx, ty); if (sp == 0.0) sp = 1.0;   // cap speed to vmax
        outx = tx / sp * vmax; outy = ty / sp * vmax;
    }
    nvx[i] = outx; nvy[i] = outy;
}

// commit new velocities and advance positions toroidally (second Python loop).
__global__ void k_apply(double *px, double *py, double *vx, double *vy,
                        const double *nvx, const double *nvy, int n,
                        double w, double h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] = nvx[i]; vy[i] = nvy[i];
    double nx = fmod(px[i] + vx[i], w); if (nx < 0) nx += w;   // Python %: always in [0,w)
    double ny = fmod(py[i] + vy[i], h); if (ny < 0) ny += h;
    px[i] = nx; py[i] = ny;
}

// ---- host helpers ----------------------------------------------------------
static double polarization(const double *vx, const double *vy, int n, double vmax) {
    double mx = 0, my = 0;
    for (int i = 0; i < n; ++i) { mx += vx[i]; my += vy[i]; }
    mx /= n; my /= n;
    return hypot(mx, my) / vmax;
}

static void render_ascii(const double *px, const double *py,
                        const double *vx, const double *vy,
                        int n, int w, int h) {
    // arrow glyphs (UTF-8 multibyte), indexed by heading octant like the Python.
    static const char *arrow[8] = {
        "→", "↗", "↑", "↖", "←", "↙", "↓", "↘"
    };
    // grid of glyph indices; -1 = empty
    int *grid = (int *)malloc((size_t)w * h * sizeof(int));
    for (int c = 0; c < w * h; ++c) grid[c] = -1;
    for (int i = 0; i < n; ++i) {
        int x = ((int)px[i] % w + w) % w;
        int y = ((int)py[i] % h + h) % h;
        double a = atan2(vy[i], vx[i]);
        int k = (int)llround(a / (M_PI / 4.0));
        k = ((k % 8) + 8) % 8;
        grid[y * w + x] = k;
    }
    printf("\nFlock (each bird points along its heading — watch them align):\n\n");
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int k = grid[y * w + x];
            if (k < 0) printf(" ");
            else printf("%s", arrow[k]);
        }
        printf("\n");
    }
    free(grid);
}

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

int main(int argc, char **argv) {
    // defaults mirror flocking.py
    int ticks = 600, n = 90, w = 90, h = 48;
    double perc = 8.0, sep_r = 3.0, wsep = 1.3, wali = 1.5, wcoh = 0.85;
    double vmax = 1.0, turn = 0.35;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--birds") || !strcmp(argv[i], "--ants")) && i + 1 < argc)
            n = atoi(argv[++i]);
    }

    // ---- host init: consume the SplitMix64 stream in EXACTLY the Python order
    // (all px, then all py, then all headings) so init matches the sequential
    // ports bit-for-bit. state = seed (cross-port convention).
    double *h_px = (double *)malloc(n * sizeof(double));
    double *h_py = (double *)malloc(n * sizeof(double));
    double *h_vx = (double *)malloc(n * sizeof(double));
    double *h_vy = (double *)malloc(n * sizeof(double));
    uint64_t st = seed;
    for (int i = 0; i < n; ++i) h_px[i] = sm64_uniform(&st, 0.0, (double)w);
    for (int i = 0; i < n; ++i) h_py[i] = sm64_uniform(&st, 0.0, (double)h);
    for (int i = 0; i < n; ++i) {
        double ang = sm64_uniform(&st, 0.0, 2.0 * M_PI);
        h_vx[i] = cos(ang); h_vy[i] = sin(ang);
    }

    double *d_px, *d_py, *d_vx, *d_vy, *d_nvx, *d_nvy;
    CK(cudaMalloc(&d_px, n * sizeof(double)));
    CK(cudaMalloc(&d_py, n * sizeof(double)));
    CK(cudaMalloc(&d_vx, n * sizeof(double)));
    CK(cudaMalloc(&d_vy, n * sizeof(double)));
    CK(cudaMalloc(&d_nvx, n * sizeof(double)));
    CK(cudaMalloc(&d_nvy, n * sizeof(double)));
    CK(cudaMemcpy(d_px, h_px, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_py, h_py, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_vx, h_vx, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_vy, h_vy, n * sizeof(double), cudaMemcpyHostToDevice));

    int blocks = (n + 127) / 128;

    // history sampling mirrors the Python: initial polarization, then every
    // ticks//12 ticks.
    int stepEvery = ticks / 12; if (stepEvery < 1) stepEvery = 1;
    double hist[64]; int nhist = 0;
    hist[nhist++] = polarization(h_vx, h_vy, n, vmax);

    for (int t = 0; t < ticks; ++t) {
        k_step<<<blocks, 128>>>(d_px, d_py, d_vx, d_vy, d_nvx, d_nvy, n,
                                (double)w, (double)h, perc, sep_r,
                                wsep, wali, wcoh, vmax, turn);
        k_apply<<<blocks, 128>>>(d_px, d_py, d_vx, d_vy, d_nvx, d_nvy, n,
                                 (double)w, (double)h);
        if (t % stepEvery == 0 && nhist < 64) {
            CK(cudaMemcpy(h_vx, d_vx, n * sizeof(double), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_vy, d_vy, n * sizeof(double), cudaMemcpyDeviceToHost));
            hist[nhist++] = polarization(h_vx, h_vy, n, vmax);
        }
    }
    CK(cudaDeviceSynchronize());

    // pull final state back for reporting
    CK(cudaMemcpy(h_px, d_px, n * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_py, d_py, n * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_vx, d_vx, n * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_vy, d_vy, n * sizeof(double), cudaMemcpyDeviceToHost));

    render_ascii(h_px, h_py, h_vx, h_vy, n, w, h);
    double pol = polarization(h_vx, h_vy, n, vmax);
    printf("\npolarization (flock alignment): %.3f  (0 = chaos, 1 = one flock)\n", pol);
    printf("polarization(t):");
    for (int i = 0; i < nhist; ++i) printf(" %.2f", hist[i]);
    printf("\n");

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_vx); cudaFree(d_vy);
    cudaFree(d_nvx); cudaFree(d_nvy);
    free(h_px); free(h_py); free(h_vx); free(h_vy);
    return 0;
}
