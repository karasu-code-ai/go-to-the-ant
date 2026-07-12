// "Go to the Ant" — a faithful Go port of Reynolds' boids flocking.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
// Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997),
// §3.5 (Birds & Fish: Flocking), after Reynolds 1987 and Heppner 1990.
//
// Ported from the authoritative Python reference (flocking.py). Behavior is
// reproduced structurally; numbers differ from CPython only because this port
// uses a shared SplitMix64 PRNG (identical across all language ports) rather
// than Python's Mersenne Twister. The five sequential ports agree with EACH
// OTHER.
//
// THE LENS (Go): each boid is an AGENT (a struct index into shared slices) and
// the flock is the SHARED STORE all agents step over. This is a step toward
// goroutine-per-agent — here the agents still step sequentially in index order
// over one mutable store, which keeps the update deterministic and identical to
// the other sequential ports. The actor/concurrency framing lives in the naming
// (Agent, shared Flock), not yet in real parallelism.
//
// Reynolds' three local rules (§3.5), preserved as provenance — these ARE the
// paper's rules:
//  1. SEPARATION — keep a minimum distance from the nearest birds (avoid collisions).
//  2. ALIGNMENT  — match velocity (speed + heading) to nearby birds.
//  3. COHESION   — stay close to the centre of the local flock.
//
// Each rule is a steering vector from the neighbours inside a perception radius;
// their weighted sum turns the bird. A single coherent, banking flock EMERGES —
// no leader, no central coordinator; each bird senses only its nearest peers.
//
// PROVENANCE: the three rules are the paper's (Reynolds' "boids"). The perception
// radius, the separation distance, and the three weights are OPERATIONALIZED —
// Parunak lists the rules but gives no numbers (Reynolds 1987 is the primary
// source for tuned constants). NO per-step randomness: the run is fully
// deterministic given the random init, so cross-port identity depends only on
// matching the init-RNG order and the neighbour-sum order (iterate j in index
// order).
package main

import (
	"flag"
	"fmt"
	"math"
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

// randomFloat in [0,1): (next() >> 11) * 2^-53
func (r *SplitMix64) randomFloat() float64 {
	return float64(r.next()>>11) * (1.0 / 9007199254740992.0)
}

// uniform(a,b): a + random_float()*(b-a)
func (r *SplitMix64) uniform(a, b float64) float64 {
	return a + r.randomFloat()*(b-a)
}

// ---- The shared store: Flock holds the boids' position/velocity slices --------

type Flock struct {
	n, w, h                      int
	px, py, vx, vy               []float64
	perc, sepR                   float64
	wsep, wali, wcoh, vmax, turn float64
}

func NewFlock(n, w, h, seed int) *Flock {
	fl := &Flock{
		n: n, w: w, h: h,
		perc: 8.0, sepR: 3.0, // OPERATIONALIZED (Reynolds 1987 tuned constants)
		wsep: 1.3, wali: 1.5, wcoh: 0.85, // OPERATIONALIZED rule weights
		vmax: 1.0, turn: 0.35,
	}
	rng := &SplitMix64{state: uint64(seed)}
	// RNG consume order MUST match the Python reference: all px, then all py,
	// then all headings. uniform(0,w) / uniform(0,h) / uniform(0,2pi).
	fl.px = make([]float64, n)
	fl.py = make([]float64, n)
	fl.vx = make([]float64, n)
	fl.vy = make([]float64, n)
	for i := 0; i < n; i++ {
		fl.px[i] = rng.uniform(0, float64(w))
	}
	for i := 0; i < n; i++ {
		fl.py[i] = rng.uniform(0, float64(h))
	}
	for i := 0; i < n; i++ {
		a := rng.uniform(0, 2*math.Pi)
		fl.vx[i] = math.Cos(a) // vx = cos(heading)
		fl.vy[i] = math.Sin(a) // vy = sin(heading)
	}
	return fl
}

// unit: normalize (x,y) to a unit vector; (0,0) below the tolerance.
func unit(x, y float64) (float64, float64) {
	m := math.Hypot(x, y)
	if m > 1e-9 {
		return x / m, y / m
	}
	return 0.0, 0.0
}

func (fl *Flock) step() float64 {
	n := fl.n
	// The three urges are computed against the CURRENT store, written into a
	// scratch velocity, then committed simultaneously (Python does nvx=vx[:]).
	nvx := make([]float64, n)
	nvy := make([]float64, n)
	copy(nvx, fl.vx)
	copy(nvy, fl.vy)
	p2 := fl.perc * fl.perc
	s2 := fl.sepR * fl.sepR
	w := float64(fl.w)
	h := float64(fl.h)
	for i := 0; i < n; i++ {
		var sx, sy, ax, ay, cx, cy float64
		cnt := 0
		for j := 0; j < n; j++ { // iterate j in index order (cross-port identity)
			if i == j {
				continue
			}
			dx := fl.px[j] - fl.px[i]
			dy := fl.py[j] - fl.py[i]
			dx -= w * math.Round(dx/w) // toroidal delta
			dy -= h * math.Round(dy/h)
			d2 := dx*dx + dy*dy
			if d2 > p2 {
				continue
			}
			cnt++
			ax += fl.vx[j] // rule 2: alignment (avg neighbour velocity)
			ay += fl.vy[j]
			cx += dx // rule 3: cohesion (toward neighbour centre)
			cy += dy
			if d2 < s2 && d2 > 1e-9 { // rule 1: separation (push from the close ones)
				sx -= dx / d2
				sy -= dy / d2
			}
		}
		if cnt > 0 {
			fc := float64(cnt)
			ax /= fc
			ay /= fc
			cx /= fc
			cy /= fc
			// NORMALIZE each urge to a unit vector so the three weights are
			// actually comparable (otherwise the position-scale cohesion vector
			// swamps the velocity-scale alignment one).
			sux, suy := unit(sx, sy)                   // rule 1: away from close birds
			aux, auy := unit(ax-fl.vx[i], ay-fl.vy[i]) // rule 2: toward neighbours' heading
			cux, cuy := unit(cx, cy)                   // rule 3: toward neighbours' centre
			accx := fl.wsep*sux + fl.wali*aux + fl.wcoh*cux
			accy := fl.wsep*suy + fl.wali*auy + fl.wcoh*cuy
			nvx[i] = fl.vx[i] + fl.turn*accx
			nvy[i] = fl.vy[i] + fl.turn*accy
			sp := math.Hypot(nvx[i], nvy[i]) // cap speed
			if sp == 0 {
				sp = 1.0
			}
			nvx[i] = nvx[i] / sp * fl.vmax
			nvy[i] = nvy[i] / sp * fl.vmax
		}
	}
	// commit: update all velocities then all positions toroidally.
	for i := 0; i < n; i++ {
		fl.vx[i] = nvx[i]
		fl.vy[i] = nvy[i]
		fl.px[i] = math.Mod(fl.px[i]+fl.vx[i], w)
		if fl.px[i] < 0 {
			fl.px[i] += w
		}
		fl.py[i] = math.Mod(fl.py[i]+fl.vy[i], h)
		if fl.py[i] < 0 {
			fl.py[i] += h
		}
	}
	return fl.polarization()
}

// polarization: order parameter |mean heading| / vmax. 0 = disordered chaos,
// 1 = one coherent flock all pointing the same way.
func (fl *Flock) polarization() float64 {
	var mx, my float64
	for i := 0; i < fl.n; i++ {
		mx += fl.vx[i]
		my += fl.vy[i]
	}
	mx /= float64(fl.n)
	my /= float64(fl.n)
	return math.Hypot(mx, my) / fl.vmax
}

// ---- Driver ------------------------------------------------------------------

func run(ticks, n, seed int) (*Flock, []float64) {
	fl := NewFlock(n, 90, 48, seed)
	hist := []float64{fl.polarization()}
	step := ticks / 12
	if step < 1 {
		step = 1
	}
	for t := 0; t < ticks; t++ {
		p := fl.step()
		if t%step == 0 {
			hist = append(hist, p)
		}
	}
	return fl, hist
}

func render(fl *Flock) {
	// arrow[k] for heading bucket k = round(atan2/(pi/4)) mod 8.
	arrow := []rune{'→', '↗', '↑', '↖', '←', '↙', '↓', '↘'}
	// one rune per grid cell (arrows are multibyte); build lines as strings.
	cell := make([][]rune, fl.h)
	for y := range cell {
		cell[y] = make([]rune, fl.w)
		for x := range cell[y] {
			cell[y][x] = ' '
		}
	}
	for i := 0; i < fl.n; i++ {
		x := ((int(fl.px[i]) % fl.w) + fl.w) % fl.w
		y := ((int(fl.py[i]) % fl.h) + fl.h) % fl.h
		a := math.Atan2(fl.vy[i], fl.vx[i])
		k := int(math.Round(a/(math.Pi/4))) % 8
		if k < 0 {
			k += 8
		}
		cell[y][x] = arrow[k]
	}
	fmt.Print("\nFlock (each bird points along its heading — watch them align):\n\n")
	for y := 0; y < fl.h; y++ {
		fmt.Println(string(cell[y]))
	}
}

func main() {
	ticks := flag.Int("ticks", 600, "number of ticks")
	birds := flag.Int("birds", 90, "number of boids")
	ants := flag.Int("ants", 0, "alias for --birds (0 = use --birds)")
	seed := flag.Int("seed", 0, "PRNG seed")
	flag.Parse()

	n := *birds
	if *ants != 0 {
		n = *ants
	}

	fl, hist := run(*ticks, n, *seed)
	render(fl)
	fmt.Printf("\npolarization (flock alignment): %.3f  (0 = chaos, 1 = one flock)\n", fl.polarization())
	fmt.Print("polarization(t):")
	for _, c := range hist {
		fmt.Printf(" %.2f", c)
	}
	fmt.Println()
}
