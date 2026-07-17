# Go to the Ant — Java port

A faithful Java port of Parunak's foraging swarm (§3.1) from
H. Van Dyke Parunak, *"'Go to the Ant': Engineering Principles from Natural
Multi-Agent Systems,"* Annals of Operations Research 75:69–101 (1997).

**Lens:** Java is the classical agent-based-modeling lineage — MASON, Repast,
and NetLogo all ran on the JVM. The port leans into that idiom: a `World` object
owns the two scalar pheromone fields, and a swarm of `Ant` objects each run a few
local rules under an explicit per-tick schedule.

Dependency-free: standard library only. The PRNG is a hand-rolled SplitMix64,
identical across all ports so the ports are directly comparable.

## Build

```
javac Forage.java
```

## Run

```
java Forage --seed 0
```

Flags (all optional): `--seed N`, `--ticks N` (default 3000), `--ants N`
(default 90), `--evap F` (default 0.015), `--wall` (add a wall with a gap).

Output: an ASCII render of the emergent `food_pher` trail, the final delivery
count, and a 20-point S-curve sample of cumulative deliveries across the run.

## Emergence signature

Deliveries stay at 0 for the first few hundred ticks (the trail must form),
then rise on an S-curve to some tens of deliveries by tick 3000 (the trail also
diffuses a little — breadth, §3.1/§4.6 — so nearby sub-trails merge). Sample runs:
seed 0 → 50 deliveries, seed 1 → 64. A pheromone trail links nest and food in
the render.

---

# Termite Nest Building — Java port

A faithful Java port of the termite construction swarm (§3.3, tracing to Kugler,
Turvey et al. 1990) from the same Parunak (1997) paper.

**Lens:** same JVM agent-based-modeling idiom — a `Mound` (the World) owns the two
shared scalar fields (`mass`, persistent structure; `scent`, decaying pheromone),
and a schedule steps a swarm of individually-instantiated `Termite` objects.

Two fields, three local rules: termites metabolize waste (the building material),
wander biased toward strong local scent, and stochastically deposit their load with
a probability that rises with local scent AND load. Each tick the scent field both
**diffuses** (a local Brownian stencil) **and decays** (§4.6), so fresh deposits at a
pile's core stay strongest — piles CLIMB into columns — while spreading lends each pile
some breadth (the arch substrate). The deposit-probability formula is OPERATIONALIZED
(the paper gives no formula, only "probability rises with local density and load").

Dependency-free: standard library only, same SplitMix64 PRNG as the foraging port.

## Build

```
javac Termites.java
```

## Run

```
java Termites --seed 0
```

Flags (all optional): `--seed N`, `--ticks N` (default 40000), `--termites N`
(alias `--ants`, default 70), `--decay F` (default 0.02).

Output: an ASCII render of the top-down mass density (columns emerge as bright
cores), the final distinct-column count and tallest-column mass, and a 12-point
sample of `columns(t)` across the run.

## Emergence signature

Scattered dabs self-concentrate into a HANDFUL of distinct columns (~4–10), one
very tall (tallest mass in the tens of thousands). The column count spikes early
as noise, then settles as columns compete and merge. Sample runs (the shared
SplitMix64 PRNG makes these **bit-identical to the sibling C/Rust/Go/JS ports**;
they differ from the Python/Mersenne-Twister reference by design):
seed 0 → 5 columns, tallest ≈ 103360; seed 1 → 4 columns, tallest ≈ 56810.

---

# Brood sorting (§3.2) — `Sort.java`

A faithful port of Deneubourg et al.'s ant brood/corpse sorting, from the same
Parunak (1997) survey (§3.2). Grid 40×24; 90 items each of types A/B/C (270
total) scattered at random; 40 ants; short memory = 15; k+ = 0.1, k- = 0.3 (Deneubourg 1991);
120000 ticks. Each ant wanders (dx,dy ∈ {-1,0,1}, toroidal), records every cell
it visits (empties included) into a bounded memory, and picks up / puts down
stochastically per the paper's two probability formulas
(`p(pickup)=(k+/(k++f))^2`, `p(putdown)=(f/(k-+f))^2`, f = fraction of memory
holding the same type). Sorting **emerges**; no ant compares the whole nest.

**Lens:** the JVM ABM idiom — a `Nest` object owns the shared item grid, and a
schedule steps individually-instantiated `SortAnt` objects each tick.

## Build

```
javac Sort.java
```

## Run

```
java Sort --seed 0
```

Flags (all optional): `--seed N`, `--ticks N` (default 120000), `--ants N`
(default 40).

Output: ASCII before/after grids, a 12-point clustering(t) sample, and the final
clustering. Clustering = mean fraction of the 8 toroidal neighbours that share
an item's type.

## Emergence signature

Initial clustering ≈ 0.33–0.35 (random scatter), rising monotonically to
≈ 0.85–0.92 as like items coalesce into single-type clumps. Sample runs:
seed 0 → 0.355 → 0.877, seed 1 → 0.327 → 0.895.

---

# Wolves: Surrounding Prey (§3.6) — `Wolves.java`

A faithful port of Korf's (1992) wolf-pack pursuit, from the same Parunak (1997)
survey (§3.6). World 80×44 continuous plane; 6 wolves; moose speed vm=0.6, wolf
speed vw=1.0; repulsion k=1.12; 260 ticks. The moose starts at centre; wolves are
uniform-random. Each agent considers 24 direction candidates at its own speed plus
staying put; the moose picks the in-bounds candidate MAXIMISING the min distance to
any wolf, and each wolf picks the candidate MINIMISING `S = d(moose) - k·d(nearest
other wolf)`. No per-step randomness — deterministic given the init.

**Lens:** the JVM ABM idiom — a `Hunt` (World) object owns the shared state (the
moose position and the wolf-position array), and an explicit per-tick schedule runs
the moose rule, then commits all six wolves' choices simultaneously.

**Provenance:** the two local rules and the score `S = d(moose) - k·d(wolf)` are
VERBATIM from the paper. The continuous plane + 24-candidate search (vs. the paper's
hex grid), the speeds, and k=1.12 are OPERATIONALIZED. The simultaneous wolf update
(each wolf reads the OLD other-wolf positions within a tick) is a language-forced
serialization matching the Python reference and is tagged OPERATIONALIZED in-source.

## Build

```
javac Wolves.java
```

## Run

```
java Wolves --seed 0
```

Flags (all optional): `--seed N`, `--ticks N` (default 260), `--wolves N`
(alias `--ants`, default 6).

Output: an ASCII M/W board, the largest escape gap around the moose, the final
nearest-wolf distance, and a 12-point `gap°(t)` sample across the run.

## Emergence signature

The pack closes and PINS the moose: the nearest-wolf distance collapses to ~0.1–0.3,
and the largest angular gap shrinks from wide, then oscillates as the ring squeezes
against the moose pressed to an edge. The emergence is the pinning/encirclement
pressure, not a perfectly even ring. Sample runs (this port's SplitMix64 PRNG, so
numbers differ from the Python reference by design): seed 0 → gap ≈ 198°, nearest
wolf 0.3; seed 1 → gap ≈ 111°, nearest wolf 0.1.

---

# Wasp Task Differentiation (§3.4) — `Wasps.java`

A faithful port of Theraulaz et al.'s *Polistes* wasp model, from the same
Parunak (1997) survey (§3.4). n=80 genetically-identical wasps; h=1.1; hf=3.0;
quantum=0.10; appetite=0.075·n; xi=0.02; phi=0.012; mob=1.6; leak=0.004;
gen=0.005; SIGMAX=4.0; 4000 ticks. Each wasp carries two scalars — Force
(mobility) and Threshold (brood sensitivity). Per tick, in order: (1) n/3
face-offs where force flows loser→winner with the Fermi probability
`p=1/(1+e^(h·(Fi−Fj)))`; (2) a force entropy-leak `F=max(0, F·(1−leak)+gen)`;
(3) foraging where a wasp near the brood forages with `p=1/(1+e^(hf·(σ−D)))`,
learning (σ−=xi) if it forages else forgetting (σ+=phi), and the brood demand
updates `D=max(0, D+appetite−W)`. Three castes **emerge**; no wasp computes the
proportions.

**Lens:** the JVM ABM idiom — a `Colony` object owns the shared population and
the brood demand `D`, and a schedule steps individually-instantiated `Wasp`
objects each tick.

**Provenance:** the two Fermi formulas are PAPER VERBATIM. The genuine §4.6 entropy
leak is Rule 1's conservative force TRANSFER; the force-relaxation bound (leak/gen
replacing an ad-hoc cap) is a SEPARATE inference beyond Parunak. The LOCAL
`dominance=(F/seenmax)^4` spatiality proxy — each wasp's own fading memory of the top
force faced, no global max — restores the Chief's high threshold. All OPERATIONALIZED —
see the header comment in `Wasps.java`.

## Build

```
javac Wasps.java
```

## Run

```
java Wasps --seed 0
```

Flags (all optional): `--seed N`, `--ticks N` (default 4000), `--wasps N`
(alias `--ants`, default 80).

Output: the emergent caste table (n, mean Force, mean Threshold per caste), the
`(Force, Threshold)` ASCII landscape, and a 13-point Forager/Nurse split(t)
sample across the run.

## Emergence signature

From 80 genetically-identical wasps, THREE castes self-separate: exactly **1
Chief** (high force ~9–10, HIGH threshold ~4), a small **Forager** band (~3–5,
force ~5, threshold ~0), and a **Nurse** majority (~75, force ~1). Chief force
≫ population mean. Sample runs (bit-identical to the sibling C/Rust/Go/JS ports
via the shared SplitMix64; they differ from the Python/MT reference by design):
seed 0 → Chief F=9.78 σ=4.00, 5 Foragers F≈5.15, 74 Nurses F≈0.88; seed 1 →
Chief F=9.59 σ=4.00, 4 Foragers F≈5.75, 75 Nurses F≈0.90.

---

# Flocking (§3.5) — `Flocking.java`

A faithful port of Reynolds' "boids" flocking (Reynolds 1987, Heppner 1990), from
the same Parunak (1997) survey (§3.5). n=90 birds; world 90×48 (toroidal);
perception radius 8; separation distance 3; weights wsep=1.3, wali=1.5, wcoh=0.85;
vmax=1.0; turn=0.35; 600 ticks. Init: positions uniform in the box, heading uniform
(vx=cos, vy=sin). Each of the three steering urges (separation, alignment, cohesion)
is NORMALIZED to a unit vector before the weighted sum, then the velocity is turned
and speed-capped. NO per-step randomness — fully deterministic given the random init.

**Lens:** the JVM ABM idiom — a `World` object owns the shared state arrays
(`px, py, vx, vy`), and a two-phase schedule steps individually-instantiated `Boid`
objects: every boid computes its next velocity from the frozen current state, then
the World commits velocities and moves positions toroidally.

**Provenance:** the three rules are Reynolds'; the perception radius, separation
distance, and three weights are OPERATIONALIZED (Parunak lists the rules but gives
no numbers — Reynolds 1987 is the primary source for the tuned constants).

## Build

```
javac Flocking.java
```

## Run

```
java Flocking --seed 0
```

Flags (all optional): `--seed N`, `--ticks N` (default 600), `--birds N`
(alias `--ants`, default 90).

Output: an ASCII arrow field (each bird points along its heading), the final
polarization, and a 12-point sample of polarization(t).

## Emergence signature

Polarization (|mean heading| / vmax) starts near 0 (~0.03–0.14, a disordered
scatter) and rises to ~0.9 as the birds fall into one coherent, banking flock.
Sample runs (this port's SplitMix64 PRNG, so numbers differ from the Python
reference by design): seed 0 → 0.08 → 0.934, seed 1 → 0.14 → 0.915.
