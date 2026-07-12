/* "Go to the Ant" - Wasp Task Differentiation, C port.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
 * discussing Theraulaz et al. 1991 (Polistes wasp task differentiation, §3.4).
 *
 * THE LENS (C): the colony state is a raw pair of arrays (F[], sig[]); the
 * mechanism laid utterly bare with manual memory and no abstraction between
 * agent and field. Every rule is index arithmetic over flat double arrays.
 *
 * Mature Polistes wasps -- genetically IDENTICAL -- split into a single Chief,
 * a band of Foragers, and a band of Nurses, with no HR department and no wasp
 * computing the proportion. Three interacting rules:
 *   1. FACE-OFFS. When two wasps meet, j beats i with the Fermi probability
 *      p = 1/(1 + e^(h*(F_i - F_j))). A quantum of Force passes loser -> winner.
 *   2. BROOD DEMAND.  D(t) = D(t-1) + appetite - W, W = food-work by foragers.
 *   3. FORAGE?  A wasp near the brood forages with p = 1/(1 + e^(hf*(sig_j - D))).
 *      Foraging LOWERS its threshold sig by xi (learning); not foraging RAISES
 *      it by phi (forgetting). Force is MOBILITY: a low-force wasp is stimulated
 *      by the brood but cannot travel to hunt.
 *
 * The joint (Force, Threshold) distribution self-separates into three castes:
 *   - Foragers  = high force, low threshold  (strong enough to move + sensitive)
 *   - Nurses    = low force,  low threshold  (attentive, but stuck near brood)
 *   - Chief     = one wasp, high force, high threshold (grounds the scales)
 *
 * PROVENANCE (carried from the Python reference):
 *   - The two Fermi formulas are PAPER VERBATIM (§3.4), tagged below.
 *   - The entropy-leak force bound (leak/gen, replacing an ad-hoc force cap) is
 *     OPERATIONALIZED (§4.6): a steady leak+gen bounds the hierarchy naturally
 *     (equilibrium mean ~ gen/leak). Honest tradeoff: dissipating force couples
 *     into the demand/threshold balance, so the Chief's high-threshold detail
 *     regresses vs a capped version.
 *   - dominance = (F/Fmax)^4 is OPERATIONALIZED as a spatiality proxy: the
 *     paper's Chief wanders and faces off (NOT near the brood), so it is rarely
 *     stimulated and its threshold drifts HIGH. (F/Fmax)^4 ~= 1 only for the
 *     single top wasp, leaving foragers (F ~ 0.7*Fmax) almost untouched, which
 *     restores the Chief's high-sigma caste.
 *
 * Dependency-free: C standard library only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* --- SplitMix64 PRNG (identical across all ports, seeded from --seed) ---
 * This is a CROSS-PORT CONVENTION; it intentionally does NOT reproduce CPython's
 * random module. All ports agree with EACH OTHER by consuming the RNG in the
 * same order, not with Python. */
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
static inline double rng_uniform(double a, double b) {
    return a + rng_float() * (b - a);
}

/* --- Colony constants (Theraulaz §3.4 parameterisation) --- */
#define SIGMAX 4.0

/* --- The colony state: raw arrays, the mechanism laid bare. --- */
static double F[4096];    /* force (mobility) per wasp */
static double sig[4096];  /* foraging threshold per wasp */

/* comparator for qsort on doubles (ascending), used only for medians */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    long ticks = 4000, n = 80;
    uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) ticks = strtol(argv[++i], NULL, 10);
        /* Python names this flag --wasps; we accept --ants (the shared CLI name) too. */
        else if ((!strcmp(argv[i], "--wasps") || !strcmp(argv[i], "--ants")) && i + 1 < argc)
            n = strtol(argv[++i], NULL, 10);
    }
    if (n < 1) n = 1;
    if (n > 4096) n = 4096;
    rng_state = seed;

    /* colony parameters (match the Python defaults) */
    const double h = 1.1, hf = 3.0, quantum = 0.10;
    const double appetite = 0.075 * (double)n;
    const double xi = 0.02, phi = 0.012, mob = 1.6;
    const double leak = 0.004, gen = 0.005;
    double D = 2.0;

    /* genetically identical: tiny initial spread only.
     * RNG order matches Python exactly: all n F-inits, then all n sig-inits. */
    for (long k = 0; k < n; k++) F[k] = 1.0 + rng_uniform(-0.05, 0.05);
    for (long k = 0; k < n; k++) sig[k] = 1.6 + rng_uniform(-0.05, 0.05);

    /* Forager/Nurse split history, sampled like the Python (every ticks//12). */
    long sample = ticks / 12; if (sample < 1) sample = 1;
    int hist_f[64], hist_n[64], nhist = 0;

    for (long t = 0; t < ticks; t++) {
        /* rule 1: face-offs -- gentle, capped, so a graded hierarchy forms. */
        for (long f = 0; f < n / 3; f++) {
            long i = (long)rng_randrange((uint64_t)n);
            long j = (long)rng_randrange((uint64_t)n);
            if (i == j) continue;
            /* PAPER §3.4 VERBATIM: p = 1/(1 + e^(h*(F_i - F_j))) */
            double pj = 1.0 / (1.0 + exp(h * (F[i] - F[j])));
            long w, l;
            if (rng_float() < pj) { w = j; l = i; } else { w = i; l = j; }
            double q = quantum < F[l] ? quantum : F[l];
            F[w] += q; F[l] -= q;                 /* force conserved in the face-off (paper) */
        }
        /* ENTROPY LEAK (§4.6, OPERATIONALIZED): force dissipates and regenerates.
         * A steady leak+gen bounds the hierarchy naturally (mean ~ gen/leak), so
         * no ad-hoc force cap is needed to stop one super-wasp. */
        for (long k = 0; k < n; k++) {
            double v = F[k] * (1.0 - leak) + gen;
            F[k] = v > 0.0 ? v : 0.0;
        }
        /* rules 2 & 3: brood stimulation + foraging. Work = COUNT of mobile foragers. */
        double Fmax = F[0];
        for (long k = 1; k < n; k++) if (F[k] > Fmax) Fmax = F[k];
        if (Fmax <= 0.0) Fmax = 1.0;
        long W = 0;
        for (long k = 0; k < n; k++) {
            /* PAPER §3.4 VERBATIM: p = 1/(1 + e^(hf*(sig - D))) */
            double pf = 1.0 / (1.0 + exp(hf * (sig[k] - D)));
            double ratio = F[k] / Fmax;
            /* SPATIALITY PROXY (OPERATIONALIZED): dom ~= 1 only for the single top wasp */
            double dom = ratio * ratio * ratio * ratio;
            if (rng_float() < pf * (1.0 - dom)) {  /* stimulated AND not away dominating */
                double s = sig[k] - xi;            /* learns: threshold drops */
                sig[k] = s > 0.0 ? s : 0.0;
                if (F[k] > mob) W++;               /* mobile enough to actually hunt */
            } else {
                double s = sig[k] + phi;           /* forgets: threshold rises */
                sig[k] = s < SIGMAX ? s : SIGMAX;
            }
        }
        D = D + appetite - (double)W;
        if (D < 0.0) D = 0.0;

        if (t % sample == 0 && nhist < 64) {
            /* count Forager/Nurse under the current caste rule */
            double sorted_sig[4096];
            memcpy(sorted_sig, sig, sizeof(double) * (size_t)n);
            qsort(sorted_sig, (size_t)n, sizeof(double), cmp_double);
            double smed = sorted_sig[n / 2];
            long chief = 0;
            for (long k = 1; k < n; k++) if (F[k] > F[chief]) chief = k;
            int nf = 0, nn = 0;
            for (long k = 0; k < n; k++) {
                if (k == chief) continue;
                if (F[k] > mob && sig[k] <= smed) nf++;
                else nn++;
            }
            hist_f[nhist] = nf; hist_n[nhist] = nn; nhist++;
        }
    }

    /* --- Final castes --- */
    long chief = 0;
    for (long k = 1; k < n; k++) if (F[k] > F[chief]) chief = k;
    double sorted_sig[4096];
    memcpy(sorted_sig, sig, sizeof(double) * (size_t)n);
    qsort(sorted_sig, (size_t)n, sizeof(double), cmp_double);
    double smed = sorted_sig[n / 2];

    /* per-caste tallies: Chief (the argmax wasp), Forager, Nurse */
    int cnt[3] = {0, 0, 0};
    double sumF[3] = {0, 0, 0}, sumS[3] = {0, 0, 0};
    double popF = 0.0;
    for (long k = 0; k < n; k++) {
        popF += F[k];
        int g;
        if (k == chief) g = 0;
        else if (F[k] > mob && sig[k] <= smed) g = 1;  /* Forager: mobile + responsive */
        else g = 2;                                     /* Nurse: immobile or unresponsive */
        cnt[g]++; sumF[g] += F[k]; sumS[g] += sig[k];
    }

    printf("Emergent castes from %ld genetically identical wasps (%ld ticks):\n\n", n, ticks);
    const char *names[3] = {"Chief", "Forager", "Nurse"};
    for (int g = 0; g < 3; g++) {
        if (cnt[g] == 0) continue;
        printf("  %-8s n=%3d   mean Force %5.2f   mean Threshold %5.2f\n",
               names[g], cnt[g], sumF[g] / cnt[g], sumS[g] / cnt[g]);
    }
    printf("\n  Chief force %.2f (pop mean %.2f), threshold %.2f\n",
           F[chief], popF / (double)n, sig[chief]);
    printf("  Forager/Nurse split(t):");
    for (int i = 0; i < nhist; i++) printf(" %d/%d", hist_f[i], hist_n[i]);
    printf("\n");

    /* --- (F,sigma) ASCII landscape: x = Force ->, y = Threshold ^ --- */
    const int cols = 48, rows = 16;
    double fmn = F[0], fmx = F[0], smn = sig[0], smx = sig[0];
    for (long k = 1; k < n; k++) {
        if (F[k] < fmn) fmn = F[k];
        if (F[k] > fmx) fmx = F[k];
        if (sig[k] < smn) smn = sig[k];
        if (sig[k] > smx) smx = sig[k];
    }
    double sorted_F[4096];
    memcpy(sorted_F, F, sizeof(double) * (size_t)n);
    qsort(sorted_F, (size_t)n, sizeof(double), cmp_double);
    double fmed = sorted_F[n / 2];
    char grid[16][48];
    memset(grid, ' ', sizeof grid);
    for (long k = 0; k < n; k++) {
        int x = (int)((F[k] - fmn) / (fmx - fmn + 1e-9) * (cols - 1));
        int y = (int)((sig[k] - smn) / (smx - smn + 1e-9) * (rows - 1));
        if (x < 0) x = 0;
        if (x > cols - 1) x = cols - 1;
        if (y < 0) y = 0;
        if (y > rows - 1) y = rows - 1;
        char mark = (k == chief) ? 'C' : (F[k] >= fmed ? 'F' : 'n');
        grid[rows - 1 - y][x] = mark;
    }
    printf("\n  (F,sigma) landscape - x = Force ->, y = Threshold ^ | C chief, F forager, n nurse:\n\n");
    for (int r = 0; r < rows; r++) {
        printf("   ");
        for (int col = 0; col < cols; col++) putchar(grid[r][col]);
        putchar('\n');
    }
    return 0;
}
