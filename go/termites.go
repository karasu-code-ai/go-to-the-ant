// "Go to the Ant" §3.3 — Termite Nest Building (Kugler/Turvey; Parunak 1997), a
// faithful Go port of the authoritative Python reference (termites.py).
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997), §3.3.
//
// Tropical termites raise metre-high mounds — columns, arches, floors — with no
// chief engineer. Parunak's three local rules (§3.3):
//   1. Metabolize bodily waste, which contains pheromone. The waste IS the building
//      material.
//   2. Wander randomly, but prefer the direction of the strongest local pheromone.
//   3. Each step, decide stochastically whether to deposit the current load.
//      p(deposit) rises with the LOCAL pheromone density AND the amount carried. A
//      full termite drops even with no nearby deposit; a termite in a very high
//      local concentration drops even a small load.
// Because pheromone DECAYS, the freshest deposits (the centre of a growing pile)
// smell strongest, so piles climb upward into COLUMNS rather than spreading. No
// termite plans the mound.
//
// THE LENS (Go): termites are AGENTS (structs) stepping sequentially over one
// shared store (the Mound, holding two fields). The actor/goroutine framing lives
// in the naming (Termite agents, shared Mound), not yet in real parallelism —
// sequential stepping keeps the update deterministic and bit-identical to the
// other sequential ports.
//
// TWO fields (top-down 2D): mass = persistent structural mass (what you see);
// scent = decaying pheromone (what biases wandering). Emergence = scattered dabs
// self-concentrate into a HANDFUL of tall columns.
//
// Numbers differ from CPython only because this port uses the shared SplitMix64
// PRNG (identical across all language ports) rather than Python's Mersenne Twister;
// the emergence is the same.
package main

import (
	"flag"
	"fmt"
)

// ---- SplitMix64 PRNG (shared across all ports; standard library only) --------
// CROSS-PORT CONVENTION: these helpers intentionally do NOT reproduce CPython's
// random module; the goal is that all sequential ports agree with EACH OTHER.

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

// randrange(n) -> int in [0,n): next() % n
func (r *SplitMix64) randrange(n int) int {
	return int(r.next() % uint64(n))
}

// DIRS: the 8 neighbours in THIS order (matches the reference exactly).
var DIRS = [8][2]int{
	{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1},
}

// ---- The shared store: Mound holds the two fields ----------------------------

type Mound struct {
	w, h  int
	rng   *SplitMix64
	mass  [][]float64 // persistent structure (viz)
	scent [][]float64 // dissipative pheromone (bias): diffuses AND decays
	buf   [][]float64 // double-buffer for the diffusion pass
	decay float64
	d     float64 // diffusion rate (Brownian spreading, §4.6)
}

func newGrid(h, w int) [][]float64 {
	g := make([][]float64, h)
	for y := range g {
		g[y] = make([]float64, w)
	}
	return g
}

func NewMound(w, h, seed int, decay float64) *Mound {
	return &Mound{
		w: w, h: h,
		rng:   &SplitMix64{state: uint64(seed)},
		mass:  newGrid(h, w),
		scent: newGrid(h, w),
		buf:   newGrid(h, w),
		decay: decay,
		d:     0.010,
	}
}

// fieldStep — the field law (§4.6 entropy leak): scent DIFFUSES (Brownian spreading,
// a local nearest-neighbour stencil) then EVAPORATES. Spreading gives each pile breadth
// — the substrate for column skirts and inter-column arches — while fresh cores still
// out-smell the spread. Double-buffered so every cell reads the OLD scent; draws NO rng,
// so cross-port bit-identity is preserved. new = old + D*(mean8 - old); then *= (1-decay).
func (m *Mound) fieldStep() {
	keep := 1.0 - m.decay
	for y := 0; y < m.h; y++ {
		for x := 0; x < m.w; x++ {
			acc := 0.0
			for _, d := range DIRS {
				nx := (x + d[0] + m.w) % m.w
				ny := (y + d[1] + m.h) % m.h
				acc += m.scent[ny][nx]
			}
			c := m.scent[y][x]
			m.buf[y][x] = (c + m.d*(acc/8.0-c)) * keep
		}
	}
	m.scent, m.buf = m.buf, m.scent
}

// columns: count distinct COLUMNS = toroidal local maxima above a fraction of the
// tallest peak. Returns (count, peak).
func (m *Mound) columns() (int, float64) {
	peak := 0.0
	for y := 0; y < m.h; y++ {
		for x := 0; x < m.w; x++ {
			if m.mass[y][x] > peak {
				peak = m.mass[y][x]
			}
		}
	}
	if peak <= 0 {
		return 0, 0.0
	}
	cut := peak * 0.15
	cnt := 0
	for y := 0; y < m.h; y++ {
		for x := 0; x < m.w; x++ {
			v := m.mass[y][x]
			if v < cut {
				continue
			}
			isMax := true
			for _, d := range DIRS {
				nx := (x + d[0] + m.w) % m.w
				ny := (y + d[1] + m.h) % m.h
				if v < m.mass[ny][nx] {
					isMax = false
					break
				}
			}
			if isMax {
				cnt++
			}
		}
	}
	return cnt, peak
}

// ---- The agent: a Termite that reads/writes the shared store -----------------

type Termite struct {
	m       *Mound
	x, y    int
	load    float64
	metab   float64
	maxload float64
}

func NewTermite(m *Mound, metab, maxload float64) *Termite {
	// RNG order matches the reference: x = randrange(w), then y = randrange(h).
	x := m.rng.randrange(m.w)
	y := m.rng.randrange(m.h)
	return &Termite{m: m, x: x, y: y, load: 0.0, metab: metab, maxload: maxload}
}

func (t *Termite) step() {
	m := t.m
	// rule 1: metabolize -> waste accumulates (capped at maxload).
	t.load = t.load + t.metab
	if t.load > t.maxload {
		t.load = t.maxload
	}
	// rule 2: wander, biased toward the strongest local scent over 8 toroidal
	// neighbours, weighted by (1 + scent*3).
	var wts [8]float64
	var nxs, nys [8]int
	tot := 0.0
	for i, d := range DIRS {
		nx := (t.x + d[0] + m.w) % m.w
		ny := (t.y + d[1] + m.h) % m.h
		w := 1.0 + m.scent[ny][nx]*3.0
		wts[i] = w
		nxs[i] = nx
		nys[i] = ny
		tot += w
	}
	r := m.rng.randomFloat() * tot
	for i := 0; i < 8; i++ {
		r -= wts[i]
		if r <= 0 {
			t.x = nxs[i]
			t.y = nys[i]
			break
		}
	}
	// rule 3: stochastic deposit — probability rises with local scent AND load; a
	// full termite always drops.
	local := m.scent[t.y][t.x]
	// OPERATIONALIZED: paper §3.3 gives NO formula, only "prob rises with local
	// density AND load". This deposit-probability formula is our operationalization.
	p := 0.01 + 0.55*(t.load/t.maxload) + 0.20*local
	if p > 1.0 {
		p = 1.0
	}
	// SHORT-CIRCUIT preserved from the reference: when load >= maxload the RNG is
	// NOT consumed (Python's `or` short-circuits), which keeps the stream in sync.
	if t.load >= t.maxload || m.rng.randomFloat() < p {
		m.mass[t.y][t.x] += t.load
		m.scent[t.y][t.x] += t.load
		t.load = 0.0
	}
}

// ---- Driver ------------------------------------------------------------------

func run(ticks, n, seed int, decay float64, verbose bool) (*Mound, []int) {
	metab, maxload := 0.4, 6.0
	mound := NewMound(58, 34, seed, decay)
	termites := make([]*Termite, n)
	for i := 0; i < n; i++ {
		termites[i] = NewTermite(mound, metab, maxload)
	}
	step := ticks / 12
	if step < 1 {
		step = 1
	}
	var hist []int
	for tk := 0; tk < ticks; tk++ {
		for _, tm := range termites {
			tm.step()
		}
		mound.fieldStep()
		if tk%step == 0 {
			c, _ := mound.columns()
			hist = append(hist, c)
		}
	}
	if verbose {
		render(mound)
		cnt, peak := mound.columns()
		fmt.Printf("\ndistinct columns (local maxima): %d | tallest column mass: %.0f\n", cnt, peak)
		fmt.Print("columns(t):")
		for _, c := range hist {
			fmt.Printf(" %d", c)
		}
		fmt.Println()
	}
	return mound, hist
}

func render(m *Mound) {
	peak := 0.0
	for y := 0; y < m.h; y++ {
		for x := 0; x < m.w; x++ {
			if m.mass[y][x] > peak {
				peak = m.mass[y][x]
			}
		}
	}
	if peak <= 0 {
		peak = 1.0
	}
	shades := " .:-=+*#%@"
	n := len(shades) - 1
	fmt.Print("\nTermite mound (top-down mass density — columns emerge as bright cores):\n\n")
	for y := 0; y < m.h; y++ {
		line := make([]byte, m.w)
		for x := 0; x < m.w; x++ {
			lvl := int(m.mass[y][x] / peak * float64(n))
			if lvl < 0 {
				lvl = 0
			}
			if lvl > n {
				lvl = n
			}
			line[x] = shades[lvl]
		}
		fmt.Println(string(line))
	}
}

func main() {
	ticks := flag.Int("ticks", 40000, "number of ticks")
	ants := flag.Int("ants", 70, "number of termites")
	decay := flag.Float64("decay", 0.02, "scent evaporation rate")
	seed := flag.Int("seed", 0, "PRNG seed")
	flag.Parse()

	run(*ticks, *ants, *seed, *decay, true)
}
