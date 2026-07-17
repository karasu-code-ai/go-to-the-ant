// "Go to the Ant" — a faithful Go port of Parunak's foraging swarm.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997), §3.1.
//
// Ported from the authoritative Python reference (go_to_the_ant.py). Behavior is
// reproduced structurally; numbers differ only because this port uses a shared
// SplitMix64 PRNG (identical across all language ports) rather than Python's
// Mersenne Twister.
//
// THE LENS (Go): ants are AGENTS and the pheromone field is the SHARED STORE.
// This is a step toward goroutine-per-agent — here the agents still step
// sequentially over a single mutable store (the World), which keeps the update
// deterministic and identical to the reference. The concurrency/actor framing is
// in the naming (Agent, shared Field), not yet in real parallelism.
//
// The five local ant rules (§3.1 "Ants: Path planning"), preserved as provenance:
//   1. Avoid obstacles.
//   2. Wander randomly, biased toward nearby pheromone (Brownian floor + local scent).
//   3. If holding food, drop pheromone at a CONSTANT RATE while walking.
//   4. If at food and not holding any, pick it up.
//   5. If at the nest and carrying food, drop it.
// Plus the field law: pheromone EVAPORATES every tick (the entropy leak), so paths
// to depleted sources — and paths laid by ants who never got home — fade. No ant
// plans a route; the network EMERGES from deposit + evaporation + weighted-random
// following.
//
// PRINCIPLE APPLIED: TWO local pheromone fields, no global homing beacon.
//   food_pher: laid by CARRIERS, followed by SEARCHERS.
//   home_pher: emitted+diffused by the NEST, followed by CARRIERS.
// Every agent senses only local fields — no agent knows where the nest is; a
// carrier just climbs the local home gradient ("communication through the
// environment").
package main

import (
	"flag"
	"fmt"
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

// DIRS: the 8 neighbours in THIS order (matches the reference exactly).
var DIRS = [8][2]int{
	{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1},
}

// ---- The shared store: World holds the two pheromone fields ------------------

type World struct {
	w, h     int
	rng      *SplitMix64
	foodPher [][]float64 // laid by carriers; followed by searchers
	homePher [][]float64 // laid by the nest (diffused); followed by carriers
	obstacle [][]bool
	nest     [2]int
	food     [2]int
	region   int
	foodQty  int64
	gap      int
	useWall  bool

	deliveries int
}

func newGrid[T any](h, w int) [][]T {
	g := make([][]T, h)
	for y := range g {
		g[y] = make([]T, w)
	}
	return g
}

func NewWorld(w, h, seed, region int, useWall bool) *World {
	wd := &World{
		w: w, h: h,
		rng:      &SplitMix64{state: uint64(seed)},
		foodPher: newGrid[float64](h, w),
		homePher: newGrid[float64](h, w),
		obstacle: newGrid[bool](h, w),
		nest:     [2]int{5, h / 2},
		food:     [2]int{w - 6, h / 2},
		region:   region,
		foodQty:  1000000000, // effectively unlimited source
		useWall:  useWall,
	}
	if useWall { // optional wall with a single gap -> the swarm must ROUTE
		wallx := w / 2
		// randint(4, h-5) inclusive in Python; kept only if --wall requested.
		span := (h - 5) - 4 + 1
		wd.gap = 4 + int(wd.rng.next()%uint64(span))
		for y := 0; y < h; y++ {
			if abs(y-wd.gap) > 2 {
				wd.obstacle[y][wallx] = true
			}
		}
	}
	return wd
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func (wd *World) free(x, y int) bool {
	return x >= 0 && x < wd.w && y >= 0 && y < wd.h && !wd.obstacle[y][x]
}

func (wd *World) atFood(x, y int) bool {
	return abs(x-wd.food[0]) <= wd.region && abs(y-wd.food[1]) <= wd.region
}

func (wd *World) atNest(x, y int) bool {
	return abs(x-wd.nest[0]) <= wd.region && abs(y-wd.nest[1]) <= wd.region
}

// evaporate: BOTH fields dissipate every tick (the entropy leak).
func (wd *World) evaporate(rate float64) {
	keep := 1.0 - rate
	for y := 0; y < wd.h; y++ {
		fr := wd.foodPher[y]
		hr := wd.homePher[y]
		for x := 0; x < wd.w; x++ {
			fr[x] *= keep
			hr[x] *= keep
		}
	}
}

// emitAndDiffuseHome: the NEST is a home-pheromone SOURCE; the marker DIFFUSES
// outward into a gradient that points home from everywhere. Carriers read only the
// LOCAL gradient — no global nest-direction. One simultaneous (Jacobi) pass.
func (wd *World) emitAndDiffuseHome() {
	nx, ny := wd.nest[0], wd.nest[1]
	for dy := -wd.region; dy <= wd.region; dy++ {
		for dx := -wd.region; dx <= wd.region; dx++ {
			x, y := nx+dx, ny+dy
			if wd.free(x, y) {
				wd.homePher[y][x] += 6.0
			}
		}
	}
	nxt := newGrid[float64](wd.h, wd.w)
	for y := 0; y < wd.h; y++ {
		copy(nxt[y], wd.homePher[y]) // non-free cells stay unchanged
	}
	for y := 0; y < wd.h; y++ {
		for x := 0; x < wd.w; x++ {
			if !wd.free(x, y) {
				continue
			}
			s := wd.homePher[y][x]
			c := 1
			for _, d := range DIRS {
				xx, yy := x+d[0], y+d[1]
				if wd.free(xx, yy) {
					s += wd.homePher[yy][xx]
					c++
				}
			}
			nxt[y][x] = s / float64(c)
		}
	}
	wd.homePher = nxt
}

// diffuseFood: the food trail SPREADS a little (Brownian breadth, §3.1/§4.6): nearby
// sub-trails "merge together into a trace." Local stencil over FREE neighbours only
// (obstacle-aware, non-toroidal); draws no rng. new = old + D*(mean_free_nbrs - old).
func (wd *World) diffuseFood(D float64) {
	if D <= 0.0 {
		return
	}
	nxt := newGrid[float64](wd.h, wd.w)
	for y := 0; y < wd.h; y++ {
		copy(nxt[y], wd.foodPher[y]) // non-free cells stay unchanged
	}
	for y := 0; y < wd.h; y++ {
		for x := 0; x < wd.w; x++ {
			if !wd.free(x, y) {
				continue
			}
			s := 0.0
			c := 0
			for _, d := range DIRS {
				xx, yy := x+d[0], y+d[1]
				if wd.free(xx, yy) {
					s += wd.foodPher[yy][xx]
					c++
				}
			}
			if c > 0 {
				cur := wd.foodPher[y][x]
				nxt[y][x] = cur + D*(s/float64(c)-cur)
			}
		}
	}
	wd.foodPher = nxt
}

// ---- The agent: an Ant that reads/writes the shared store --------------------

type Agent struct {
	w        *World
	x, y     int
	carrying bool
}

func NewAgent(w *World) *Agent {
	return &Agent{w: w, x: w.nest[0], y: w.nest[1], carrying: false}
}

// weights over the 8 DIRS. Rule 2, fully LOCAL: follow the field that leads where
// you're going. Searchers read the FOOD scent; carriers read the HOME scent.
// Brownian floor (the 1.0) keeps the walk alive even on a strong trail.
func (a *Agent) weights() [8]float64 {
	var field [][]float64
	if a.carrying {
		field = a.w.homePher
	} else {
		field = a.w.foodPher
	}
	var wts [8]float64
	for i, d := range DIRS {
		nx, ny := a.x+d[0], a.y+d[1]
		if !a.w.free(nx, ny) { // rule 1: never step into a wall / off-grid
			wts[i] = 0.0
			continue
		}
		wts[i] = 1.0 + field[ny][nx]*6.0
	}
	return wts
}

func (a *Agent) step(deposit float64) {
	wts := a.weights()
	tot := 0.0
	for _, w := range wts {
		tot += w
	}
	if tot <= 0 { // boxed in — stay put this tick
		return
	}
	r := a.w.rng.randomFloat() * tot
	acc := 0.0
	for i, d := range DIRS {
		acc += wts[i]
		if r <= acc { // note the <=, matches the reference
			a.x += d[0]
			a.y += d[1]
			break
		}
	}
	// rule 3: carriers lay the FOOD trail (searchers lay nothing; the nest
	// broadcasts the HOME field instead).
	if a.carrying {
		a.w.foodPher[a.y][a.x] += deposit
	}
	// rule 4: pick up food
	if a.w.atFood(a.x, a.y) && !a.carrying && a.w.foodQty > 0 {
		a.carrying = true
		a.w.foodQty--
	} else if a.w.atNest(a.x, a.y) && a.carrying { // rule 5: drop at the nest
		a.carrying = false
		a.w.deliveries++
	}
}

// ---- Driver ------------------------------------------------------------------

type sample struct {
	t, d int
}

func run(ticks, nAnts int, evap, deposit float64, seed int, useWall bool, diffuse float64) (*World, []Agent, []sample) {
	world := NewWorld(56, 28, seed, 2, useWall)
	agents := make([]Agent, nAnts)
	for i := range agents {
		agents[i] = *NewAgent(world)
	}
	step := ticks / 20
	if step < 1 {
		step = 1
	}
	var history []sample
	for t := 0; t < ticks; t++ {
		world.emitAndDiffuseHome() // nest broadcasts the home gradient
		for i := range agents {
			agents[i].step(deposit)
		}
		world.diffuseFood(diffuse) // the food trail spreads a little (breadth)
		world.evaporate(evap)
		if t%step == 0 {
			history = append(history, sample{t, world.deliveries})
		}
	}
	return world, agents, history
}

func renderASCII(world *World, agents []Agent) {
	peak := 0.0
	for y := 0; y < world.h; y++ {
		for x := 0; x < world.w; x++ {
			if world.foodPher[y][x] > peak {
				peak = world.foodPher[y][x]
			}
		}
	}
	if peak == 0.0 {
		peak = 1.0
	}
	shades := " .:-=+*#%@"
	antpos := make(map[[2]int]bool)
	for i := range agents {
		antpos[[2]int{agents[i].x, agents[i].y}] = true
	}
	fmt.Print("\nGo to the Ant — emergent trail (nest N <-> food F, wall '|', pheromone density):\n\n")
	for y := 0; y < world.h; y++ {
		line := make([]byte, world.w)
		for x := 0; x < world.w; x++ {
			var c byte
			switch {
			case x == world.nest[0] && y == world.nest[1]:
				c = 'N'
			case x == world.food[0] && y == world.food[1]:
				c = 'F'
			case world.obstacle[y][x]:
				c = '|'
			case antpos[[2]int{x, y}]:
				c = 'o'
			default:
				lvl := int((world.foodPher[y][x] / peak) * float64(len(shades)-1))
				if lvl < 0 {
					lvl = 0
				}
				if lvl > len(shades)-1 {
					lvl = len(shades) - 1
				}
				c = shades[lvl]
			}
			line[x] = c
		}
		fmt.Println(string(line))
	}
}

func main() {
	ticks := flag.Int("ticks", 3000, "number of ticks")
	ants := flag.Int("ants", 90, "number of ants")
	evap := flag.Float64("evap", 0.015, "evaporation rate")
	deposit := flag.Float64("deposit", 1.0, "pheromone deposit per carrier step")
	seed := flag.Int("seed", 0, "PRNG seed")
	useWall := flag.Bool("wall", false, "add a wall with a gap (the routing demo)")
	diffuse := flag.Float64("diffuse", 0.03, "food-trail diffusion (breadth); 0 = off")
	flag.Parse()

	world, agents, hist := run(*ticks, *ants, *evap, *deposit, *seed, *useWall, *diffuse)
	renderASCII(world, agents)
	fmt.Printf("\nfood delivered to nest over %d ticks: %d\n", *ticks, world.deliveries)
	fmt.Print("deliveries(t): ")
	for i, s := range hist {
		if i > 0 {
			fmt.Print(" ")
		}
		fmt.Printf("%d", s.d)
	}
	fmt.Println()
}
