/* "Go to the Ant" — birds & fish: flocking (Reynolds 1987), C port.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
 * §3.5. Primary source for the three rules and tuned constants: C. W. Reynolds,
 * "Flocks, Herds, and Schools: A Distributed Behavioral Model," SIGGRAPH 1987.
 *
 * THE LENS (C): the state is a raw array of doubles — px/py/vx/vy, four flat
 * lanes with manual memory. The flocking mechanism laid utterly bare: no boid
 * object, no neighbour abstraction, just an O(n^2) index sweep summing steering
 * urges directly out of the position/velocity arrays.
 *
 * Reynolds' three local rules (§3.5), each a steering vector from the neighbours
 * inside a perception radius; their weighted sum turns the bird:
 *   1. SEPARATION — keep a minimum distance from the nearest birds.
 *   2. ALIGNMENT  — match velocity (speed + heading) to nearby birds.
 *   3. COHESION   — stay close to the centre of the local flock.
 * Global coordination (one coherent, banking flock) EMERGES from these three
 * local urges — no leader, no central plan.
 *
 * PROVENANCE: the three rules are the paper's (Reynolds' "boids"). The
 * perception radius, the separation distance, and the three weights are
 * OPERATIONALIZED — Parunak lists the rules but gives no numbers (Reynolds 1987
 * is the primary source for tuned constants).
 *
 * NO per-step randomness — fully deterministic given the random init. Cross-port
 * identity depends only on matching the init-RNG order and the neighbour-sum
 * order (iterate j in index order).
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
static inline double rng_uniform(double a, double b) { /* [a,b) */
    return a + rng_float() * (b - a);
}

/* --- World / flock parameters (OPERATIONALIZED, Reynolds 1987) --- */
#define N_DEFAULT 90
#define W 90
#define H 48
static const double PERC  = 8.0;    /* perception radius */
static const double SEP_R = 3.0;    /* separation distance */
static const double WSEP  = 1.3;    /* separation weight */
static const double WALI  = 1.5;    /* alignment weight */
static const double WCOH  = 0.85;   /* cohesion weight */
static const double VMAX  = 1.0;    /* speed cap */
static const double TURN  = 0.35;   /* steering gain */

/* --- The flock: raw parallel arrays. --- */
static double px[N_DEFAULT], py[N_DEFAULT];   /* positions */
static double vx[N_DEFAULT], vy[N_DEFAULT];   /* velocities (unit-ish, |v|=vmax) */
static double nvx[N_DEFAULT], nvy[N_DEFAULT]; /* next-tick velocities (Jacobi) */

/* Normalize (x,y) to a unit vector; (0,0) if too small. */
static inline void unit(double x, double y, double *ox, double *oy) {
    double m = hypot(x, y);
    if (m > 1e-9) { *ox = x / m; *oy = y / m; }
    else          { *ox = 0.0;   *oy = 0.0;   }
}

/* One synchronous tick over n birds. Returns polarization after the update. */
static double step(int n) {
    double p2 = PERC * PERC, s2 = SEP_R * SEP_R;
    for (int i = 0; i < n; i++) {
        double sx = 0, sy = 0, ax = 0, ay = 0, cx = 0, cy = 0;
        int cnt = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double dx = px[j] - px[i], dy = py[j] - py[i];
            /* toroidal delta: rint = round-half-to-even, matching Python round() */
            dx -= (double)W * rint(dx / (double)W);
            dy -= (double)H * rint(dy / (double)H);
            double d2 = dx * dx + dy * dy;
            if (d2 > p2) continue;
            cnt++;
            ax += vx[j]; ay += vy[j];           /* rule 2: alignment (avg neighbour velocity) */
            cx += dx;    cy += dy;               /* rule 3: cohesion (toward neighbour centre) */
            if (d2 < s2 && d2 > 1e-9) {          /* rule 1: separation (push from the close ones) */
                sx -= dx / d2; sy -= dy / d2;
            }
        }
        if (cnt) {
            ax /= cnt; ay /= cnt; cx /= cnt; cy /= cnt;
            /* NORMALIZE each urge to a unit vector so the three weights are
             * actually comparable (else the position-scale cohesion vector
             * swamps the velocity-scale alignment one). */
            double sux, suy, aux, auy, cux, cuy;
            unit(sx, sy, &sux, &suy);                 /* rule 1: away from close birds */
            unit(ax - vx[i], ay - vy[i], &aux, &auy); /* rule 2: toward neighbours' heading */
            unit(cx, cy, &cux, &cuy);                 /* rule 3: toward neighbours' centre */
            double accx = WSEP * sux + WALI * aux + WCOH * cux;
            double accy = WSEP * suy + WALI * auy + WCOH * cuy;
            double nx = vx[i] + TURN * accx, ny = vy[i] + TURN * accy;
            double sp = hypot(nx, ny); if (sp == 0.0) sp = 1.0; /* cap speed */
            nvx[i] = nx / sp * VMAX; nvy[i] = ny / sp * VMAX;
        } else {
            nvx[i] = vx[i]; nvy[i] = vy[i];
        }
    }
    /* commit velocities, then advance positions toroidally */
    for (int i = 0; i < n; i++) {
        vx[i] = nvx[i]; vy[i] = nvy[i];
        px[i] = fmod(px[i] + vx[i], (double)W); if (px[i] < 0) px[i] += (double)W;
        py[i] = fmod(py[i] + vy[i], (double)H); if (py[i] < 0) py[i] += (double)H;
    }
    /* polarization: |mean heading| / vmax */
    double mx = 0, my = 0;
    for (int i = 0; i < n; i++) { mx += vx[i]; my += vy[i]; }
    mx /= n; my /= n;
    return hypot(mx, my) / VMAX;
}

static double polarization(int n) {
    double mx = 0, my = 0;
    for (int i = 0; i < n; i++) { mx += vx[i]; my += vy[i]; }
    mx /= n; my /= n;
    return hypot(mx, my) / VMAX;
}

/* ASCII arrow field — each bird points along its heading. */
static void render(int n) {
    static char grid[H][W];
    memset(grid, ' ', sizeof grid);
    /* arrow[k] for k = round(atan2(vy,vx)/(pi/4)) mod 8, matching Python */
    static const char *arrow[8] = {
        "\xe2\x86\x92", /* -> */ "\xe2\x86\x97", /* NE */ "\xe2\x86\x91", /* up */
        "\xe2\x86\x96", /* NW */ "\xe2\x86\x90", /* <- */ "\xe2\x86\x99", /* SW */
        "\xe2\x86\x93", /* down */ "\xe2\x86\x98" /* SE */
    };
    static int cellk[H][W];
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { grid[y][x] = ' '; cellk[y][x] = -1; }
    for (int i = 0; i < n; i++) {
        int x = (int)px[i] % W, y = (int)py[i] % H;
        if (x < 0) x += W;
        if (y < 0) y += H;
        double a = atan2(vy[i], vx[i]);
        int k = ((int)rint(a / (M_PI / 4.0))) % 8; if (k < 0) k += 8;
        cellk[y][x] = k;
    }
    printf("\nFlock (each bird points along its heading — watch them align):\n\n");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (cellk[y][x] >= 0) fputs(arrow[cellk[y][x]], stdout);
            else putchar(' ');
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    long ticks = 600, n = N_DEFAULT;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = strtol(argv[++i], NULL, 10);
        else if ((!strcmp(argv[i], "--birds") || !strcmp(argv[i], "--ants")) && i + 1 < argc)
            n = strtol(argv[++i], NULL, 10);
    }
    if (n > N_DEFAULT) n = N_DEFAULT; /* fixed-size arrays sized for the default flock */
    rng_state = seed;

    /* Init — consume the RNG in EXACTLY the Python order:
     * px[0..n) uniform(0,W), then py[0..n) uniform(0,H), then ang[0..n) uniform(0,2pi). */
    for (int i = 0; i < n; i++) px[i] = rng_uniform(0.0, (double)W);
    for (int i = 0; i < n; i++) py[i] = rng_uniform(0.0, (double)H);
    for (int i = 0; i < n; i++) {
        double a = rng_uniform(0.0, 2.0 * M_PI);
        vx[i] = cos(a); vy[i] = sin(a);
    }

    /* polarization(t) sample: initial value, then every ticks//12 ticks at t=0.. */
    long every = ticks / 12; if (every < 1) every = 1;
    double hist[64]; int nhist = 0;
    hist[nhist++] = polarization((int)n);
    for (long t = 0; t < ticks; t++) {
        double p = step((int)n);
        if (t % every == 0 && nhist < 64) hist[nhist++] = p;
    }

    render((int)n);
    printf("\npolarization (flock alignment): %.3f  (0 = chaos, 1 = one flock)\n",
           polarization((int)n));
    printf("polarization(t):");
    for (int i = 0; i < nhist; i++) printf(" %.2f", hist[i]);
    printf("\n");
    return 0;
}
