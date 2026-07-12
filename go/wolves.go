// "Go to the Ant" §3.6 — Wolves: Surrounding Prey (Korf 1992), a faithful Go port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
// §3.6, after R. E. Korf, "A simple solution to pursuit games" (1992).
//
// Ported from the authoritative Python reference (wolves.py). Behavior is
// reproduced structurally; numbers differ from Python only because this port
// uses the shared SplitMix64 PRNG (identical across all language ports) rather
// than Python's Mersenne Twister.
//
// THE LENS (Go): each wolf is an AGENT (a struct) and the world is the SHARED
// STORE all agents step over. This is a step toward goroutine-per-agent — here
// the agents still step sequentially over one mutable store (the Hunt), which
// keeps the update deterministic and identical to the reference. The
// actor/concurrency framing lives in the naming (Agent structs, one shared
// Hunt), not yet in real parallelism.
//
// One wolf can't kill a moose; the pack must SURROUND it — with no radios and no
// negotiated strategy. Parunak gives two local rules (§3.6):
//   1. MOOSE: move to the neighbouring point FARTHEST from the nearest wolf.
//   2. WOLVES: move to minimise  S = d(moose) - k*d(nearest other wolf)  — i.e.
//      get CLOSE to the moose while staying FAR from each other. k tunes the
//      wolf-wolf repulsion. Attraction (to prey) + repulsion (between wolves)
//      balanced => the pack encircles the moose, no communication required.
//
// PROVENANCE: the rules and the score  S = d(moose) - k*d(wolf)  are the paper's
// (Korf 1992) — VERBATIM. The speeds, k=1.12, and the continuous-plane /
// 24-candidate-direction search (vs. the paper's six wolves on a hex grid) are
// OPERATIONALIZED (the paper is qualitative on the geometry).
package main

import (
	"flag"
	"fmt"
	"math"
	"sort"
)

const TAU = 2 * math.Pi

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

// uniform(a,b): a + random_float()*(b-a)
func (r *SplitMix64) uniform(a, b float64) float64 {
	return a + r.randomFloat()*(b-a)
}

// ---- The shared store: the Hunt (moose + wolf positions) ---------------------

type vec struct{ x, y float64 }

// Hunt is the single mutable store all agents read/step over.
type Hunt struct {
	w, h       float64
	vm, vw, k  float64
	rng        *SplitMix64
	mx, my     float64 // moose position
	wolves     []vec   // wolf agents
}

func NewHunt(nWolves, w, h, seed int, vm, vw, k float64) *Hunt {
	rng := &SplitMix64{state: uint64(seed)}
	hunt := &Hunt{
		w: float64(w), h: float64(h),
		vm: vm, vw: vw, k: k,
		rng: rng,
		mx:  float64(w) / 2, my: float64(h) / 2, // moose starts at centre
	}
	// Wolves uniform-random. Consume the RNG in the SAME order as Python:
	// per wolf, x = uniform(0,w) THEN y = uniform(0,h).
	hunt.wolves = make([]vec, nWolves)
	for i := 0; i < nWolves; i++ {
		x := rng.uniform(0, float64(w))
		y := rng.uniform(0, float64(h))
		hunt.wolves[i] = vec{x, y}
	}
	return hunt
}

// cands: the 24 directional candidate moves for an agent at (x,y) with speed sp,
// PLUS the stay-put point (x,y) appended last (matches the reference order).
func (hunt *Hunt) cands(x, y, sp float64) []vec {
	out := make([]vec, 0, 25)
	for i := 0; i < 24; i++ {
		a := float64(i) * TAU / 24
		out = append(out, vec{x + sp*math.Cos(a), y + sp*math.Sin(a)})
	}
	out = append(out, vec{x, y})
	return out
}

func (hunt *Hunt) inBounds(x, y float64) bool {
	return 0 <= x && x < hunt.w && 0 <= y && y < hunt.h
}

// step advances the whole hunt one tick and returns the new gap().
func (hunt *Hunt) step() float64 {
	W := hunt.wolves
	// rule 1: the moose flees to the in-bounds candidate FARTHEST from its
	// nearest wolf (maximise the min distance to any wolf). Strict > keeps the
	// first-seen candidate on ties, matching Python.
	best := vec{hunt.mx, hunt.my}
	bd := -1.0
	for _, c := range hunt.cands(hunt.mx, hunt.my, hunt.vm) {
		if !hunt.inBounds(c.x, c.y) {
			continue
		}
		d := math.Inf(1)
		for _, wv := range W {
			dd := math.Hypot(c.x-wv.x, c.y-wv.y)
			if dd < d {
				d = dd
			}
		}
		if d > bd {
			bd, best = d, c
		}
	}
	hunt.mx, hunt.my = best.x, best.y

	// rule 2: each wolf minimises  S = d(moose) - k*d(nearest OTHER wolf).
	// OPERATIONALIZED update order: wolves are scored against the OTHER wolves'
	// positions from the START of this tick (Python builds a fresh list `nw`
	// and assigns it only after all wolves are decided). This is a serialized /
	// simultaneous face-off, not per-wolf sequential — reported as a deviation.
	nw := make([]vec, len(W))
	for i, wv := range W {
		bestW := vec{wv.x, wv.y}
		bs := 1e9
		for _, c := range hunt.cands(wv.x, wv.y, hunt.vw) {
			if !hunt.inBounds(c.x, c.y) {
				continue
			}
			dm := math.Hypot(c.x-hunt.mx, c.y-hunt.my)
			// d(nearest OTHER wolf); default 0.0 if there is no other wolf.
			do := 0.0
			first := true
			for j, ov := range W {
				if j == i {
					continue
				}
				dd := math.Hypot(c.x-ov.x, c.y-ov.y)
				if first || dd < do {
					do, first = dd, false
				}
			}
			s := dm - hunt.k*do // S = d(moose) - k*d(wolf)  [VERBATIM]
			if s < bs {
				bs, bestW = s, c
			}
		}
		nw[i] = bestW
	}
	hunt.wolves = nw
	return hunt.gap()
}

// gap: the largest angular gap (deg) between adjacent wolves as seen from the
// moose. 360/N when evenly ringed => surrounded; near 360 when all on one side.
func (hunt *Hunt) gap() float64 {
	n := len(hunt.wolves)
	if n < 2 {
		return 360.0
	}
	angs := make([]float64, n)
	for i, wv := range hunt.wolves {
		angs[i] = math.Atan2(wv.y-hunt.my, wv.x-hunt.mx)
	}
	sort.Float64s(angs)
	maxGap := 0.0
	for i := 0; i < n; i++ {
		g := math.Mod(angs[(i+1)%n]-angs[i], TAU)
		if g < 0 {
			g += TAU
		}
		if g > maxGap {
			maxGap = g
		}
	}
	return maxGap * 180 / math.Pi
}

func (hunt *Hunt) nearestWolf() float64 {
	md := math.Inf(1)
	for _, wv := range hunt.wolves {
		d := math.Hypot(hunt.mx-wv.x, hunt.my-wv.y)
		if d < md {
			md = d
		}
	}
	return md
}

// ---- Driver ------------------------------------------------------------------

func run(ticks, nWolves, seed int) (*Hunt, []float64) {
	hunt := NewHunt(nWolves, 80, 44, seed, 0.6, 1.0, 1.12)
	hist := []float64{hunt.gap()}
	sampleEvery := ticks / 12
	if sampleEvery < 1 {
		sampleEvery = 1
	}
	for t := 0; t < ticks; t++ {
		g := hunt.step()
		if t%sampleEvery == 0 {
			hist = append(hist, g)
		}
	}
	return hunt, hist
}

func render(hunt *Hunt) {
	w, h := int(hunt.w), int(hunt.h)
	grid := make([][]byte, h)
	for y := range grid {
		grid[y] = make([]byte, w)
		for x := range grid[y] {
			grid[y][x] = ' '
		}
	}
	mod := func(a, m int) int { return ((a % m) + m) % m }
	for _, wv := range hunt.wolves {
		grid[mod(int(wv.y), h)][mod(int(wv.x), w)] = 'W'
	}
	grid[mod(int(hunt.my), h)][mod(int(hunt.mx), w)] = 'M'
	fmt.Print("\nThe hunt (M moose, W wolves — watch the ring close):\n\n")
	for _, row := range grid {
		fmt.Println(string(row))
	}
}

func main() {
	ticks := flag.Int("ticks", 260, "number of ticks")
	wolves := flag.Int("wolves", 6, "number of wolves")
	ants := flag.Int("ants", 6, "alias for --wolves (agent count)")
	seed := flag.Int("seed", 0, "PRNG seed")
	flag.Parse()

	nWolves := *wolves
	// If --ants is passed non-default, honour it as the agent-count alias.
	if *ants != 6 {
		nWolves = *ants
	}

	hunt, hist := run(*ticks, nWolves, *seed)
	render(hunt)
	fmt.Printf("\nlargest escape gap around the moose: %.0f°  (evenly surrounded ≈ %d°) | nearest wolf %.1f\n",
		hunt.gap(), 360/nWolves, hunt.nearestWolf())
	fmt.Print("gap°(t):")
	for _, g := range hist {
		fmt.Printf(" %.0f", g)
	}
	fmt.Println()
}
