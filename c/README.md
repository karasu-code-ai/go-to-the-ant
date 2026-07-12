# Go to the Ant — C port

A faithful C port of Parunak's foraging swarm (Parunak, *"'Go to the Ant':
Engineering Principles from Natural Multi-Agent Systems"*, Annals of Operations
Research 75:69-101, 1997, §3.1).

**The lens (C):** the pheromone field is a raw shared array of `double`s —
stigmergy laid utterly bare. Manual memory, a hand-rolled SplitMix64 PRNG, flat
index arithmetic. The hardcore baseline the other ports are measured against.

Two local pheromone fields (communication through the environment, no global
nest-direction): `food_pher` laid by carriers and followed by searchers;
`home_pher` emitted and diffused by the nest and followed by carriers.
Dependency-free — C standard library only.

## Build

```sh
gcc -O2 -Wall -Wextra -std=c11 -o forage forage.c -lm
```

## Run

```sh
./forage --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (3000), `--ants N` (90),
`--evap F` (0.015). Prints the final delivery count, a 20-point S-curve sample,
and an ASCII render of the `food_pher` trail.

## Emergence signature

With 90 ants over 3000 ticks: deliveries stay at 0 for the first few hundred
ticks (the trail must form first), then rise on an S-curve to some tens of
deliveries, with a visible pheromone trail connecting nest and food.
Observed: seed 0 -> 46, seed 1 -> 60.

---

# Ant brood sorting (§3.2)

A faithful C port of Parunak's brood-sorting swarm (§3.2, after Deneubourg et
al. 1991). Grid 40x24; 90 items each of types A/B/C (270 total) scattered by a
shuffle; 40 ants with a 10-cell memory; k+ = 1, k- = 3; 120000 ticks.

**The lens (C):** the nest is a raw `unsigned char` array — one byte per cell,
0 for empty or a type tag A/B/C. An ant is three ints plus a ring buffer of the
last 10 cells it saw. Pickup and putdown are direct byte writes into the shared
grid; the sorting mechanism laid utterly bare, no abstraction between agent and
field.

Ants wander (toroidal), record every cell (incl. empties) into short memory,
pick up an item with `p = (k+/(k+ + f))^2` and drop a carried item on empty
ground with `p = (f/(k- + f))^2`, where `f` is the fraction of memory holding
the same type (both formulas paper-verbatim, §3.2). Clustering = mean fraction
of the 8 toroidal neighbours that share an item's type.

## Build

```sh
gcc -O2 -Wall -Wextra -std=c11 -o sort sort.c -lm
```

## Run

```sh
./sort --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (120000), `--ants N` (40). Prints the
before/after ASCII grids, the initial and final clustering, and a 12-point
clustering(t) sample.

## Emergence signature

Initial clustering ~0.33 (a random scatter), rising monotonically to ~0.85-0.92
as like items coalesce into single-type clusters.
Observed: seed 0 -> 0.263 -> 0.876, seed 1 -> 0.268 -> 0.903 (seeds 0/1 happen
to draw slightly low initial scatters; seeds 2-5 start ~0.33).

---

# Go to the Ant — termites (C port)

A faithful C port of Parunak's termite nest building (§3.3, after
Kugler/Turvey; Kugler et al. 1990).

**The lens (C):** the state is a raw array — the mechanism laid utterly bare
with manual memory, no abstraction between agent and field. Two flat `double`
fields — `mass[]` (persistent structure, what you see) and `scent[]` (decaying
pheromone, what biases wandering) — plus a struct-of-scalars termite on a
toroidal grid. Dependency-free — C standard library only.

Three local rules: metabolize waste into carried load; wander over the 8
toroidal neighbours weighted by `1 + scent*3`; stochastically deposit with
`p = min(1, 0.01 + 0.55*(load/maxload) + 0.20*local_scent)` (a full termite
always drops). `scent` evaporates each tick, so the freshest core of a pile
smells strongest and piles climb into columns. The deposit formula is
OPERATIONALIZED — the paper gives only "probability rises with local density
AND load", no formula.

## Build

```sh
gcc -O2 -std=c11 -o termites termites.c -lm
```

## Run

```sh
./termites --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (40000), `--termites N` / `--ants N`
(70), `--decay F` (0.02). Prints an ASCII mass render, `columns(t)`, and the
final distinct-column count and tallest column mass.

## Emergence signature

Scattered dabs self-concentrate into a HANDFUL of distinct columns (~4-10),
one very tall (tallest mass in the tens of thousands). `columns(t)` rises to
~13-14 early then descends as piles merge into a few winners.
Observed (SplitMix64 stream, matches the sibling ports, not Python's RNG):
seed 0 -> 5 columns, tallest 96053; seed 1 -> 4 columns, tallest 133850.

---

# Go to the Ant — wasps (C port)

A faithful C port of Parunak's wasp task differentiation (§3.4, after
Theraulaz et al. 1991; Polistes wasps).

**The lens (C):** the colony state is a raw pair of arrays — `F[]` (force /
mobility) and `sig[]` (foraging threshold) — the mechanism laid utterly bare
with manual memory and no abstraction between agent and field. Every rule is
index arithmetic over flat `double` arrays. Dependency-free — C standard
library only.

Three interacting rules, applied per tick in this order: (1) `n/3` face-offs
where `j` beats `i` with Fermi `p = 1/(1 + e^(h*(F_i - F_j)))` and a force
quantum passes loser->winner (force conserved); (2) an entropy leak
`F = max(0, F*(1-leak) + gen)` that bounds the hierarchy naturally
(OPERATIONALIZED, §4.6 — replaces an ad-hoc force cap); (3) brood stimulation +
foraging with Fermi `p = 1/(1 + e^(hf*(sig - D)))`, suppressed for the top wasp
by a spatiality proxy `dom = (F/Fmax)^4` (OPERATIONALIZED). Both Fermi formulas
are PAPER VERBATIM.

## Build

```sh
gcc -O2 -std=c11 -o wasps wasps.c -lm
```

## Run

```sh
./wasps --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (4000), `--wasps N` / `--ants N`
(80). Prints the caste table (n, mean force, mean threshold per caste), the
`(F,sigma)` ASCII landscape, and the Forager/Nurse split(t).

## Emergence signature

From 80 genetically IDENTICAL wasps, THREE castes self-separate: exactly 1
Chief (high force ~9-10, HIGH threshold ~4), a small band of Foragers (~2-8,
force ~5, threshold ~0), and a Nurse majority (~70+, force ~1). Chief force is
far above the population mean.
Observed (SplitMix64 stream, matches the sibling ports, not Python's RNG):
seed 0 -> Chief F=9.78 sigma=4.00, 4 Foragers F~4.88, 75 Nurses F~0.94
(pop mean 1.25); seed 1 -> Chief F=9.59 sigma=4.00, 3 Foragers F~5.52,
76 Nurses F~0.97.

---

# Go to the Ant — birds & fish: flocking (C port)

A faithful C port of Parunak's flocking swarm (§3.5, after Reynolds 1987,
Heppner 1990). World 90x48; 90 boids; perception radius 8; separation distance
3; weights sep 1.3 / align 1.5 / cohesion 0.85; vmax 1.0; turn 0.35; 600 ticks.

**The lens (C):** the state is a raw array of `double`s — px/py/vx/vy, four flat
lanes with manual memory, no boid object and no neighbour abstraction. The
flocking mechanism laid utterly bare: an O(n^2) index sweep summing the three
steering urges directly out of the position/velocity arrays.

Reynolds' three local rules, each a steering vector from the neighbours inside
the perception radius (toroidal deltas): separation (push from birds closer than
`sep_r`, summed as `-d/d2`), alignment (average neighbour velocity), cohesion
(toward the neighbour centroid). Each urge is normalized to a UNIT vector before
weighting, so `acc = wsep*sep + wali*(align-vel) + wcoh*coh`; `v += turn*acc`,
speed capped to `vmax`, positions advanced toroidally. NO per-step randomness —
fully deterministic given the random init. The rules are the paper's; the radius,
separation distance and weights are OPERATIONALIZED (Reynolds 1987 is the primary
source for tuned constants).

## Build

```sh
gcc -O2 -std=c11 -o flocking flocking.c -lm
```

## Run

```sh
./flocking --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (600), `--birds N` / `--ants N` (90).
Prints an ASCII arrow field (each bird points along its heading), the final
polarization, and a `polarization(t)` sample.

## Emergence signature

Polarization (|mean heading| / vmax) starts near 0 (~0.03-0.15, a disordered
scatter of headings) and rises to ~0.9 as the boids merge into one coherent
flock all pointing the same way.
Observed (SplitMix64 stream, matches the sibling ports, not Python's RNG):
seed 0 -> 0.08 -> 0.898, seed 1 -> 0.14 -> 0.874.

---

# Go to the Ant — wolves: surrounding prey (C port)

A faithful C port of Parunak's wolf-pack pursuit (§3.6, after Korf 1992). World
80x44; 6 wolves; moose speed vm=0.6; wolf speed vw=1.0; repulsion k=1.12; 260
ticks. The moose starts at the centre; wolves are placed uniform-random.

**The lens (C):** the state is a raw pair of float lanes (`wolf_x[]`,
`wolf_y[]`) plus the moose scalars — the mechanism laid utterly bare with manual
memory, no abstraction between agent and geometry. Each agent's move is chosen by
generating the 25 candidate points (24 compass directions at its speed, plus
staying put) by hand and scanning them with raw `hypot` loops. Dependency-free —
C standard library only.

Two local rules, applied per tick in this order: (1) the MOOSE picks the
in-bounds candidate MAXIMISING the min distance to any wolf; (2) each WOLF, in
index order, picks the candidate MINIMISING `S = d(moose) - k*d(nearest other
wolf)` — the score is PAPER VERBATIM. No per-step randomness; the hunt is
deterministic given the initial placement. `gap()` is the largest angular gap
(deg) between adjacent wolves as seen from the moose. The continuous plane and
the 24-candidate search (vs the paper's hex grid), plus vm/vw/k, are
OPERATIONALIZED. The wolves are updated sequentially reading the current
positions of the not-yet-moved wolves (mirrors the Python's read-W-write-nw
pattern) — an OPERATIONALIZED serialization noted in the source.

## Build

```sh
gcc -O2 -std=c11 -o wolves wolves.c -lm
```

## Run

```sh
./wolves --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (260), `--wolves N` (6). Prints the
ASCII M/W board, the final escape gap and nearest-wolf distance, and a
`gap deg(t)` sample.

## Emergence signature

The pack closes and PINS the moose: the nearest-wolf distance collapses to ~0.2
and the largest angular gap shrinks from wide then oscillates — the emergence is
the encirclement/pinning pressure, not a perfectly even ring.
Observed (SplitMix64 stream, matches the sibling ports, not Python's RNG):
seed 0 -> gap 198 deg, nearest wolf 0.3; seed 1 -> gap 111 deg, nearest wolf 0.1.
(Cross-checked: driving the Python reference with this same SplitMix64
initialisation reproduces seed 0 -> gap 198, nearest 0.3 exactly, confirming the
dynamics are bit-identical and only the base PRNG differs.)
