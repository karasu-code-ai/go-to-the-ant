// "Go to the Ant" — a faithful Go port of Deneubourg's ant brood sorting.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
// §3.2 (after Deneubourg et al. 1991).
//
// Ported from the authoritative Python reference (brood_sorting.py). Behavior is
// reproduced structurally; numbers agree with the OTHER language ports because
// this port uses a shared SplitMix64 PRNG (identical across all ports) rather
// than Python's Mersenne Twister.
//
// THE LENS (Go): ants are AGENTS (structs) stepping sequentially over ONE shared
// store (the Nest grid). This is a step toward goroutine-per-agent — here the
// agents still step in index order over a single mutable store, which keeps the
// update deterministic and identical to the reference. The actor/concurrency
// framing lives in the naming (SortAnt agents, shared Nest), not in real
// parallelism.
//
// The four local rules (§3.2), preserved as provenance:
//   1. Wander randomly around the nest.
//   2. Keep a SHORT memory (~15 steps) of the object types recently seen.
//   3. Not carrying + at an object: pick it up stochastically.
//        p(pickup) = (k+/(k+ + f))^2   -- PAPER §3.2 VERBATIM
//   4. Carrying + on empty ground: drop it stochastically.
//        p(putdown) = (f/(k- + f))^2   -- PAPER §3.2 VERBATIM
//   f is the fraction of short memory holding the SAME type.
//   Constants (paper): k+ 0.1, k- 0.3 (Deneubourg 1991; Parunak's summary rounds to ~1, ~3) (k- must exceed k+ or clusters dissolve
//   faster than they form). mem=15 (Deneubourg 1991).  -- these three are OPERATIONALIZED as the
//   concrete numbers the paper cites (kp=0.1, km=0.3, mem=15 (Deneubourg 1991)).
// Local concentrations of like items emerge, retain members, and attract more;
// stochastic pickup lets separate clusters merge. Sorting EMERGES; no ant
// compares the whole nest.
package main

import (
	"flag"
	"fmt"
)

// TYPES: the kinds of brood items (larvae / eggs / cocoons). 0 = empty cell.
const TYPES = "ABC"

// ---- SplitMix64 PRNG (shared across all ports; standard library only) --------

type SplitMix64 struct{ state uint64 }

func (r *SplitMix64) next() uint64 {
	r.state += 0x9E3779B97F4A7C15
	z := r.state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// randomFloat in [0,1): (next() >> 11) * 2^-53
func (r *SplitMix64) randomFloat() float64 {
	return float64(r.next()>>11) * (1.0 / 9007199254740992.0)
}

// randrange(n) -> int in [0,n)
func (r *SplitMix64) randrange(n int) int {
	return int(r.next() % uint64(n))
}

// shuffle: Fisher-Yates, matching the reference exactly (i from len-1 downto 1).
func (r *SplitMix64) shuffle(arr [][2]int) {
	for i := len(arr) - 1; i >= 1; i-- {
		j := r.randrange(i + 1)
		arr[i], arr[j] = arr[j], arr[i]
	}
}

// ---- The shared store: Nest holds the item grid ------------------------------
//
// grid stores item type as a byte: 0 means empty, otherwise 'A'/'B'/'C'.

type Nest struct {
	w, h int
	rng  *SplitMix64
	grid [][]byte
}

func NewNest(w, h, nPerType, seed int) *Nest {
	n := &Nest{w: w, h: h, rng: &SplitMix64{state: uint64(seed)}}
	n.grid = make([][]byte, h)
	for y := range n.grid {
		n.grid[y] = make([]byte, w)
	}
	// cells in (x,y) with y outer, x inner — same order as the reference.
	cells := make([][2]int, 0, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			cells = append(cells, [2]int{x, y})
		}
	}
	n.rng.shuffle(cells)
	i := 0
	for t := 0; t < len(TYPES); t++ { // scatter each type at random
		for k := 0; k < nPerType; k++ {
			x, y := cells[i][0], cells[i][1]
			i++
			n.grid[y][x] = TYPES[t]
		}
	}
	return n
}

// clustering: mean fraction of 8 toroidal neighbours that share an item's type.
// 0 = scattered, 1 = perfectly sorted. Iterates y outer, x inner, and dx/dy in
// the reference order so the metric is bit-identical across ports.
func (n *Nest) clustering() float64 {
	tot := 0
	same := 0.0
	for y := 0; y < n.h; y++ {
		for x := 0; x < n.w; x++ {
			t := n.grid[y][x]
			if t == 0 {
				continue
			}
			neigh := 0
			simt := 0
			for _, dx := range []int{-1, 0, 1} {
				for _, dy := range []int{-1, 0, 1} {
					if dx == 0 && dy == 0 {
						continue
					}
					nx := ((x+dx)%n.w + n.w) % n.w
					ny := ((y+dy)%n.h + n.h) % n.h
					if n.grid[ny][nx] != 0 {
						neigh++
						if n.grid[ny][nx] == t {
							simt++
						}
					}
				}
			}
			if neigh > 0 {
				tot++
				same += float64(simt) / float64(neigh)
			}
		}
	}
	if tot < 1 {
		tot = 1
	}
	return same / float64(tot)
}

// ---- The agent: a SortAnt that reads/writes the shared store -----------------

type SortAnt struct {
	n        *Nest
	x, y     int
	carry    byte // 0 = not carrying
	mem      []byte
	memCap   int
	kp, km   float64
}

func NewSortAnt(n *Nest, mem int, kp, km float64) *SortAnt {
	// RNG order matches the reference: randrange(w) then randrange(h).
	a := &SortAnt{n: n, kp: kp, km: km, memCap: mem}
	a.x = n.rng.randrange(n.w)
	a.y = n.rng.randrange(n.h)
	a.mem = make([]byte, 0, mem)
	return a
}

// _f: fraction of short memory holding the SAME type t.
func (a *SortAnt) f(t byte) float64 {
	if len(a.mem) == 0 {
		return 0.0
	}
	c := 0
	for _, m := range a.mem {
		if m == t {
			c++
		}
	}
	return float64(c) / float64(len(a.mem))
}

// appendMem: push onto the bounded deque (maxlen=memCap).
func (a *SortAnt) appendMem(here byte) {
	if len(a.mem) == a.memCap {
		a.mem = a.mem[1:]
	}
	a.mem = append(a.mem, here)
}

// choiceDelta: choice((-1,0,1)) == [-1,0,1][randrange(3)].
func (a *SortAnt) choiceDelta() int {
	return []int{-1, 0, 1}[a.n.rng.randrange(3)]
}

func (a *SortAnt) step() {
	n := a.n
	// rule 1: wander (toroidal). dx consumed before dy, matching the reference.
	a.x = ((a.x+a.choiceDelta())%n.w + n.w) % n.w
	a.y = ((a.y+a.choiceDelta())%n.h + n.h) % n.h
	here := n.grid[a.y][a.x]
	a.appendMem(here) // rule 2: record even empties
	if a.carry == 0 {
		if here != 0 { // rule 3: maybe pick up
			// PAPER §3.2 VERBATIM: p(pickup) = (k+/(k+ + f))^2
			p := a.kp / (a.kp + a.f(here))
			p = p * p
			if n.rng.randomFloat() < p {
				a.carry = here
				n.grid[a.y][a.x] = 0
			}
		}
	} else {
		if here == 0 { // rule 4: maybe drop
			// PAPER §3.2 VERBATIM: p(putdown) = (f/(k- + f))^2
			fc := a.f(a.carry)
			p := fc / (a.km + fc)
			p = p * p
			if n.rng.randomFloat() < p {
				n.grid[a.y][a.x] = a.carry
				a.carry = 0
			}
		}
	}
}

// ---- Rendering ---------------------------------------------------------------

func render(n *Nest) {
	for y := 0; y < n.h; y++ {
		line := make([]byte, n.w)
		for x := 0; x < n.w; x++ {
			if n.grid[y][x] == 0 {
				line[x] = '.'
			} else {
				line[x] = n.grid[y][x]
			}
		}
		fmt.Println(string(line))
	}
}

// ---- Driver ------------------------------------------------------------------

func run(ticks, nAnts, seed int, verbose bool) (*Nest, []float64) {
	nest := NewNest(40, 24, 90, seed)
	ants := make([]*SortAnt, nAnts)
	for i := range ants {
		ants[i] = NewSortAnt(nest, 15, 0.1, 0.3)
	}
	if verbose {
		fmt.Print("BEFORE (random scatter):\n\n")
		render(nest)
		fmt.Printf("\ninitial clustering: %.3f\n", nest.clustering())
	}
	sampleEvery := ticks / 12
	if sampleEvery < 1 {
		sampleEvery = 1
	}
	var hist []float64
	for t := 0; t < ticks; t++ {
		for _, a := range ants {
			a.step()
		}
		if t%sampleEvery == 0 {
			hist = append(hist, nest.clustering())
		}
	}
	if verbose {
		fmt.Print("\nAFTER (emergent sorting):\n\n")
		render(nest)
		fmt.Printf("\nfinal clustering: %.3f\n", nest.clustering())
		fmt.Print("clustering(t):")
		for _, c := range hist {
			fmt.Printf(" %.2f", c)
		}
		fmt.Println()
	}
	return nest, hist
}

func main() {
	ticks := flag.Int("ticks", 120000, "number of ticks")
	ants := flag.Int("ants", 40, "number of ants")
	seed := flag.Int("seed", 0, "PRNG seed")
	flag.Parse()

	run(*ticks, *ants, *seed, true)
}
