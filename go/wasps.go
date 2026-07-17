// "Go to the Ant" §3.4 — Wasp Task Differentiation (Theraulaz et al. 1991), a Go port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997), §3.4.
//
// Ported from the authoritative Python reference (wasps.py). Behavior is reproduced
// structurally; numbers match the other sequential ports because this port uses a
// shared SplitMix64 PRNG (identical across all language ports) rather than Python's
// Mersenne Twister — the ports agree with EACH OTHER, not with Python.
//
// THE LENS (Go): each wasp is an AGENT (a struct field-set) and the colony is the
// SHARED STORE all agents step over. This is a step toward goroutine-per-agent —
// here the agents still update sequentially over one mutable store (the Colony),
// which keeps the update deterministic and identical to the reference. The
// actor/concurrency framing lives in the naming (Colony as the shared store, wasps
// as indexed agents), not yet in real parallelism.
//
// Mature Polistes wasps — genetically IDENTICAL — split into a single Chief, a band
// of Foragers, and a band of Nurses, with no HR department and no wasp computing the
// proportion. Parunak's three interacting rules:
//  1. FACE-OFFS. When two wasps meet, j beats i with the Fermi probability
//     p = 1/(1 + e^(h·(F_i − F_j))). The higher force usually wins (but not always);
//     a quantum of Force passes loser → winner (force is conserved).
//  2. BROOD DEMAND. D(t) = D(t−1) + appetite − W, where W is the food-work done.
//  3. FORAGE? A wasp near the brood forages with p = 1/(1 + e^(hf·(σ_j − D))).
//     Foraging LOWERS its threshold σ by ξ (learning); not foraging RAISES σ by φ.
//
// Force is MOBILITY (a low-force wasp is stimulated by the brood but cannot travel to
// hunt). The joint (Force, Threshold) distribution self-separates into three castes:
//
//	· Foragers = high force, low threshold  (strong enough to move + sensitive)
//	· Nurses   = low force,  low threshold  (attentive, but stuck near the brood)
//	· Chief    = one wasp, high force, high threshold (grounds the scales; doesn't forage)
package main

import (
	"flag"
	"fmt"
	"math"
	"sort"
)

// ---- SplitMix64 PRNG (shared across all ports; standard library only) --------

type SplitMix64 struct{ state uint64 }

func (r *SplitMix64) next() uint64 {
	r.state += 0x9E3779B97F4A7C15
	z := r.state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// random_float in [0,1): (next() >> 11) * 2^-53
func (r *SplitMix64) randomFloat() float64 {
	return float64(r.next()>>11) * (1.0 / 9007199254740992.0)
}

// randrange(n) -> int in [0,n)
func (r *SplitMix64) randrange(n int) int {
	return int(r.next() % uint64(n))
}

// uniform(a,b): a + random_float()*(b-a)
func (r *SplitMix64) uniform(a, b float64) float64 {
	return a + r.randomFloat()*(b-a)
}

// ---- The shared store: the Colony all wasp-agents step over -------------------

type Colony struct {
	rng *SplitMix64
	n   int
	F   []float64 // force  (mobility) per wasp-agent
	sig []float64 // threshold per wasp-agent
	seenmax []float64 // per-wasp FADING memory of the top force it has faced (LOCAL, no global max)
	D   float64   // brood demand

	h, hf, q     float64
	appetite     float64
	xi, phi, mob float64
	leak, gen    float64
	seendecay    float64 // the seenmax memory fades (robustness)
}

const SIGMAX = 4.0

// NOTE (provenance, honest labelling): the genuine §4.6 entropy leak for wasps is RULE 1's
// conservative force TRANSFER (Parunak names "the flow of force among wasps"). The leak/gen
// term below is a SEPARATE Force-RELAXATION (mean-reversion to ~gen/leak) — an INFERENCE BEYOND
// Parunak (plausibly a Theraulaz 1991 element we can't verify), replacing an ad-hoc force CAP.
// Empirically required: pure conservation condenses to one super-wasp with no graded forager
// band. The dominance term is now LOCAL (per-wasp seenmax). See PROVENANCE.md.
func NewColony(n, seed int) *Colony {
	c := &Colony{
		rng:      &SplitMix64{state: uint64(seed)},
		n:        n,
		D:        2.0,
		h:        1.1,
		hf:       3.0,
		q:        0.10,
		appetite: 0.075 * float64(n),
		xi:       0.02,
		phi:      0.012,
		mob:      1.6,
		leak:     0.004,
		gen:      0.005,
		seendecay: 0.998,
	}
	c.F = make([]float64, n)
	c.sig = make([]float64, n)
	c.seenmax = make([]float64, n)
	// genetically identical: tiny initial spread only. Consume the RNG in EXACTLY
	// the Python order — all F inits first, then all sig inits.
	for i := 0; i < n; i++ {
		c.F[i] = 1.0 + c.rng.uniform(-0.05, 0.05)
	}
	for i := 0; i < n; i++ {
		c.sig[i] = 1.6 + c.rng.uniform(-0.05, 0.05)
	}
	copy(c.seenmax, c.F)
	return c
}

func (c *Colony) step() {
	F, sig, n := c.F, c.sig, c.n
	// rule 1: face-offs — gentle, capped, so a graded hierarchy forms (not one super-wasp)
	for k := 0; k < n/3; k++ {
		i, j := c.rng.randrange(n), c.rng.randrange(n)
		if i == j {
			continue
		}
		// PAPER §3.4 VERBATIM: p = 1/(1 + e^(h·(F_i − F_j)))
		fi, fj := F[i], F[j]
		pj := 1.0 / (1.0 + math.Exp(c.h*(fi-fj)))
		var w, l int
		if c.rng.randomFloat() < pj {
			w, l = j, i
		} else {
			w, l = i, j
		}
		t := c.q
		if F[l] < t {
			t = F[l]
		}
		F[w] += t // force is conserved in the face-off (paper)
		F[l] -= t
		m := fi // LOCAL: each wasp's FADING memory of the strongest force it faced
		if fj > m {
			m = fj
		}
		di, dj := c.seenmax[i]*c.seendecay, c.seenmax[j]*c.seendecay
		if m > di {
			c.seenmax[i] = m
		} else {
			c.seenmax[i] = di
		}
		if m > dj {
			c.seenmax[j] = m
		} else {
			c.seenmax[j] = dj
		}
	}
	// FORCE RELAXATION (inference beyond Parunak, NOT the §4.6 entropy leak — that is
	// Rule 1's conservative force flow above): force mean-reverts toward ~gen/leak each
	// tick. A steady leak+gen bounds the hierarchy naturally, so no ad-hoc force cap is
	// needed. Empirically required (pure conservation condenses to one super-wasp).
	for k := 0; k < n; k++ {
		v := F[k]*(1.0-c.leak) + c.gen
		if v < 0.0 {
			v = 0.0
		}
		F[k] = v
	}
	// rules 2 & 3: brood stimulation + foraging. Work = COUNT of mobile foragers.
	// SPATIALITY PROXY (operationalized, now LOCAL): the paper's Chief "wanders and faces
	// off", so it is NOT near the brood and is rarely stimulated -> its threshold drifts
	// HIGH. We approximate "away dominating" via dominance=(F/seenmax)^4, where seenmax is
	// each wasp's OWN fading memory of the top force it has faced (NO global max) -> ~1
	// only for the wasp atop its own encounters (the Chief). Restores its high-σ caste.
	W := 0
	for k := 0; k < n; k++ {
		// PAPER §3.4 VERBATIM: p = 1/(1 + e^(hf·(σ − D)))
		pf := 1.0 / (1.0 + math.Exp(c.hf*(sig[k]-c.D)))
		sm := c.seenmax[k]
		if sm <= 0.0 {
			sm = 1.0
		}
		ratio := F[k] / sm
		dom := ratio * ratio * ratio * ratio    // (F/seenmax)^4 ~1 only for the wasp atop its own encounters
		if c.rng.randomFloat() < pf*(1.0-dom) { // stimulated AND not away dominating
			sig[k] = sig[k] - c.xi // learns: threshold drops
			if sig[k] < 0.0 {
				sig[k] = 0.0
			}
			if F[k] > c.mob { // mobile enough to actually hunt
				W++
			}
		} else {
			sig[k] = sig[k] + c.phi // forgets: threshold rises
			if sig[k] > SIGMAX {
				sig[k] = SIGMAX
			}
		}
	}
	c.D = c.D + c.appetite - float64(W)
	if c.D < 0.0 {
		c.D = 0.0
	}
}

// castes returns the three caste index-groups and the Chief index.
func (c *Colony) castes() (chief int, forager, nurse []int) {
	chief = 0
	for k := 1; k < c.n; k++ {
		if c.F[k] > c.F[chief] {
			chief = k
		}
	}
	sorted := make([]float64, c.n)
	copy(sorted, c.sig)
	sort.Float64s(sorted)
	smed := sorted[c.n/2]
	for k := 0; k < c.n; k++ {
		if k == chief {
			continue
		} else if c.F[k] > c.mob && c.sig[k] <= smed {
			forager = append(forager, k) // mobile + responsive
		} else {
			nurse = append(nurse, k) // immobile (or unresponsive) -> stays with brood
		}
	}
	return
}

// ---- Driver ------------------------------------------------------------------

type sample struct{ f, ns int }

func run(ticks, n, seed int, verbose bool) *Colony {
	c := NewColony(n, seed)
	var hist []sample
	step := ticks / 12
	if step < 1 {
		step = 1
	}
	for t := 0; t < ticks; t++ {
		c.step()
		if t%step == 0 {
			_, fg, nu := c.castes()
			hist = append(hist, sample{len(fg), len(nu)})
		}
	}
	if verbose {
		chief, fg, nu := c.castes()
		fmt.Printf("Emergent castes from %d genetically identical wasps (%d ticks):\n\n", n, ticks)
		printCaste("Chief", []int{chief}, c)
		printCaste("Forager", fg, c)
		printCaste("Nurse", nu, c)
		var sumF float64
		for k := 0; k < n; k++ {
			sumF += c.F[k]
		}
		fmt.Printf("\n  Chief force %.2f (pop mean %.2f), threshold %.2f\n",
			c.F[chief], sumF/float64(n), c.sig[chief])
		fmt.Print("  Forager/Nurse split(t):")
		for _, s := range hist {
			fmt.Printf(" %d/%d", s.f, s.ns)
		}
		fmt.Println()
		landscape(c, 48, 16)
	}
	return c
}

func printCaste(name string, ks []int, c *Colony) {
	if len(ks) == 0 {
		return
	}
	var mF, mS float64
	for _, k := range ks {
		mF += c.F[k]
		mS += c.sig[k]
	}
	mF /= float64(len(ks))
	mS /= float64(len(ks))
	fmt.Printf("  %-8s n=%3d   mean Force %5.2f   mean Threshold %5.2f\n", name, len(ks), mF, mS)
}

// landscape: ASCII scatter of the population in (Force -> x, Threshold -> y) space.
func landscape(c *Colony, cols, rows int) {
	fmn, fmx := c.F[0], c.F[0]
	smn, smx := c.sig[0], c.sig[0]
	for k := 1; k < c.n; k++ {
		if c.F[k] < fmn {
			fmn = c.F[k]
		}
		if c.F[k] > fmx {
			fmx = c.F[k]
		}
		if c.sig[k] < smn {
			smn = c.sig[k]
		}
		if c.sig[k] > smx {
			smx = c.sig[k]
		}
	}
	chief := 0
	for k := 1; k < c.n; k++ {
		if c.F[k] > c.F[chief] {
			chief = k
		}
	}
	sortedF := make([]float64, c.n)
	copy(sortedF, c.F)
	sort.Float64s(sortedF)
	fmed := sortedF[c.n/2]

	grid := make([][]byte, rows)
	for y := range grid {
		grid[y] = make([]byte, cols)
		for x := range grid[y] {
			grid[y][x] = ' '
		}
	}
	for k := 0; k < c.n; k++ {
		x := int((c.F[k] - fmn) / (fmx - fmn + 1e-9) * float64(cols-1))
		y := int((c.sig[k] - smn) / (smx - smn + 1e-9) * float64(rows-1))
		var mark byte
		if k == chief {
			mark = 'C'
		} else if c.F[k] >= fmed {
			mark = 'F'
		} else {
			mark = 'n'
		}
		grid[rows-1-y][x] = mark
	}
	fmt.Print("\n  (F,σ) landscape — x = Force →, y = Threshold ↑ | C chief, F forager, n nurse:\n\n")
	for _, row := range grid {
		fmt.Println("   " + string(row))
	}
}

func main() {
	ticks := flag.Int("ticks", 4000, "number of ticks")
	wasps := flag.Int("wasps", 80, "number of wasps")
	ants := flag.Int("ants", 0, "alias for --wasps (colony size)")
	seed := flag.Int("seed", 0, "PRNG seed")
	flag.Parse()

	n := *wasps
	if *ants != 0 {
		n = *ants
	}
	run(*ticks, n, *seed, true)
}
