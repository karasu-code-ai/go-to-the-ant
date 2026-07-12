# Go to the Ant — Go port

A faithful Go port of Parunak's foraging swarm (Parunak 1997, §3.1), ported from
the authoritative Python reference `go_to_the_ant.py`.

**The lens (Go):** ants are **agents** and the pheromone field is the **shared
store** (the `World`). This is a step toward goroutine-per-agent — here the agents
still step sequentially over one mutable store, which keeps the update
deterministic and identical to the reference. The actor/concurrency framing lives
in the naming (`Agent`, shared `World`), not yet in real parallelism.

Two local pheromone fields (stigmergy — communication through the environment):
`foodPher` laid by carriers and followed by searchers; `homePher` emitted and
diffused by the nest and followed by carriers. No agent knows where the nest is; a
carrier just climbs the local home gradient.

Dependency-free: standard library only. Shared **SplitMix64** PRNG so all ports
are directly comparable.

## Build & run

```sh
# requires Go 1.18+ on your PATH
cd go
go build -o forage forage.go     # build
./forage --seed 0                # run

# or without an explicit build:
go run forage.go --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (3000), `--ants N` (90),
`--evap F` (0.015), `--deposit F` (1.0), `--wall` (optional wall+gap routing demo).

## Emergence signature (seed 0, 3000 ticks, 90 ants)

- deliveries stay 0 for roughly the first ~900 ticks (the trail must form first),
- then a clear S-curve rise to tens of deliveries (~46 at seed 0, ~60 at seed 1),
- and a visible `foodPher` trail connecting nest `N` and food `F` in the ASCII render.

Numbers differ from the Python reference only because of the different PRNG
(SplitMix64 vs. Mersenne Twister); the qualitative behavior is the same.

---

# Ant brood sorting — Go port

A faithful Go port of Deneubourg's ant brood sorting (Parunak 1997, §3.2, after
Deneubourg et al. 1991), ported from the authoritative Python reference
`brood_sorting.py`.

**The lens (Go):** ants are **agents** (`SortAnt` structs) stepping sequentially,
in index order, over ONE shared store (the `Nest` grid). The actor framing lives
in the naming (agents over a shared store), not in real parallelism — sequential
stepping keeps the update deterministic and identical to the reference.

Four local rules (stigmergy — communication through the environment): wander;
keep a short ~10-step memory of item types seen; not-carrying + on an item pick up
with `p=(k+/(k++f))^2`; carrying + on empty drop with `p=(f/(k-+f))^2`, where `f`
is the fraction of memory holding the SAME type and `k+=1 < k-=3`. Local clusters
of like items emerge, retain members, and attract more. No ant compares the whole
nest — sorting EMERGES.

Dependency-free: standard library only. Shared **SplitMix64** PRNG so all ports
are directly comparable.

## Build & run

```sh
# requires Go 1.18+ on your PATH
cd go
go build -o sort sort.go          # build
./sort --seed 0                   # run (full 120000 ticks)

# or without an explicit build:
go run sort.go --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (120000), `--ants N` (40).

## Emergence signature (grid 40x24, 270 items, 40 ants, 120000 ticks)

Clustering (mean fraction of the 8 toroidal neighbours sharing an item's type)
starts low and rises monotonically to the high 0.8s / low 0.9s:

- seed 0: `0.263 -> 0.876`
- seed 1: `0.268 -> 0.903`

and the AFTER grid shows visibly consolidated A/B/C patches. The initial value is
lower than the Python reference's `0.350` only because the SplitMix64 shuffle
produces a different random scatter (the ports agree with EACH OTHER, not with
Python's Mersenne Twister); the rising-to-sorted signature is identical.

---

# Wasps — Go port

A faithful Go port of Parunak's wasp task-differentiation model (Parunak 1997,
§3.4; Theraulaz et al. 1991), ported from the authoritative Python reference
`wasps.py`.

**The lens (Go):** each wasp is an **agent** (a struct field-set) and the colony
is the **shared store** all agents step over sequentially — the actor framing is
in the naming, not yet in real parallelism, which keeps the update deterministic
and identical to the reference.

Three interacting rules (two Fermi formulas preserved VERBATIM; the entropy-leak
force bound and the `dominance=(F/Fmax)^4` spatiality proxy preserved as
OPERATIONALIZED — see the provenance comments in `wasps.go`) drive genetically
identical wasps to self-separate into three castes.

## Build & run

```sh
# requires Go 1.18+ on your PATH
cd go
go build -o wasps wasps.go        # build
./wasps --seed 0                  # run (full 4000 ticks)

# or without an explicit build:
go run wasps.go --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (4000), `--wasps N` (80, also `--ants`).

## Emergence signature (seed 0, 4000 ticks, 80 wasps)

From genetically identical wasps, THREE castes emerge:

- exactly **1 Chief** — high force (~9-10), HIGH threshold (~4), force >> pop mean,
- a small band of **Foragers** (~2-8) — force ~5, low threshold ~0,
- a **Nurse** majority (~70+) — force ~1.

Output: the caste table (n, mean Force, mean Threshold per caste), the (Force,
threshold) ASCII landscape, and the Forager/Nurse split over time. Numbers agree
with the other sequential ports (shared SplitMix64), not with Python.

---

# Termite Nest Building — Go port

A faithful Go port of Parunak 1997 §3.3 (Kugler/Turvey termite mounds), ported
from the authoritative Python reference `termites.py`.

**The lens (Go):** termites are **agents** (structs) stepping sequentially over one
shared store (the `Mound`, holding two fields). The actor/goroutine framing lives
in the naming, not yet in real parallelism — sequential stepping keeps the update
bit-identical to the other sequential ports.

Two fields (stigmergy): `mass` = persistent structural mass (what you see);
`scent` = decaying pheromone (what biases wandering). Emergence = scattered dabs
self-concentrate into a handful of tall columns. The deposit-probability formula is
**OPERATIONALIZED** — the paper gives no formula, only "probability rises with local
density AND load".

## Build & run

```sh
# requires Go 1.18+ on your PATH
cd go
go build -o termites termites.go   # build (name the file — do NOT use ./...)
./termites --seed 0                # run (full 40000 ticks)

# or without an explicit build:
go run termites.go --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (40000), `--ants N` (70),
`--decay F` (0.02).

## Emergence signature (seed 0, 40000 ticks, 70 termites)

- scattered dabs self-concentrate into a HANDFUL of distinct columns (~5-10),
- one column grows very tall (tallest mass in the tens of thousands),
- `columns(t)` peaks early (many transient dabs) then settles to the handful.

Observed: seed 0 -> 5 columns, tallest mass ~96053; seed 1 -> 4 columns, tallest
~133850. (Python seed 0 gives 7 columns / ~92659; the difference is only the PRNG.)

---

# Boids Flocking — Go port

A faithful Go port of Reynolds' boids flocking (Parunak 1997, §3.5; after
Reynolds 1987, Heppner 1990), ported from the authoritative Python reference
`flocking.py`.

**The lens (Go):** each boid is an **agent** (a struct index into shared
position/velocity slices) and the flock is the **shared store** all agents step
over sequentially, in index order. The actor/goroutine framing lives in the
naming (agents over a shared `Flock`), not yet in real parallelism — sequential
stepping keeps the update deterministic and identical to the other sequential
ports.

Reynolds' three local rules (these ARE the paper's; the perception radius,
separation distance, and three weights are OPERATIONALIZED — Reynolds 1987 is the
primary source for tuned constants): **separation** (push from birds inside
`sep_r`), **alignment** (match neighbour velocity), **cohesion** (steer to the
neighbour centre). Each urge is normalized to a unit vector, weighted, and summed
into the turn. There is **no per-step randomness** — the run is fully
deterministic given the random init, so cross-port identity depends only on the
init-RNG order and the neighbour-sum order.

Dependency-free: standard library only. Shared **SplitMix64** PRNG so all ports
are directly comparable.

## Build & run

```sh
# requires Go 1.18+ on your PATH
cd go
go build -o flocking flocking.go   # build (name the file — do NOT use ./...)
./flocking --seed 0                # run (full 600 ticks)

# or without an explicit build:
go run flocking.go --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (600), `--birds N` (90, also `--ants`).

## Emergence signature (seed 0, 600 ticks, 90 boids, world 90x48)

Polarization (order parameter `|mean heading| / vmax`) starts near 0 (a random
scatter of headings) and rises to ~0.9 as one coherent flock forms:

- seed 0: `0.08 -> 0.913`
- seed 1: `0.14 -> 0.928`

Output: an ASCII arrow field (each boid points along its heading), the
`polarization(t)` curve, and the final polarization. Numbers agree with the other
sequential ports (shared SplitMix64), not with Python's Mersenne Twister; the
starts-near-0 / rises-to-one-flock signature is identical (Python seed 0:
`0.08 -> 0.917`).

---

# Wolves: Surrounding Prey — Go port

A faithful Go port of Parunak 1997 §3.6 (Korf 1992, pursuit games), ported from
the authoritative Python reference `wolves.py`.

**The lens (Go):** each wolf is an **agent** (a struct) and the world is the
**shared store** (`Hunt`) all agents step over sequentially. The actor/goroutine
framing lives in the naming, not yet in real parallelism — sequential stepping
keeps the update deterministic and identical to the reference.

Two local rules (no communication): the **moose** flees to the candidate point
farthest from its nearest wolf; each **wolf** minimises `S = d(moose) - k*d(nearest
other wolf)` — close to prey, far from packmates. `S` is **VERBATIM** (Korf 1992);
the continuous plane + 24-candidate search and `k=1.12` are **OPERATIONALIZED**
(the paper uses six wolves on a hex grid). The wolf update is a serialized
face-off (all wolves score against the tick-start positions), reported as a
language-forced deviation.

## Build & run

```sh
# requires Go 1.18+ on your PATH
cd go
go build -o wolves wolves.go      # build (name the file — do NOT use ./...)
./wolves --seed 0                 # run (full 260 ticks)

# or without an explicit build:
go run wolves.go --seed 0
```

Flags: `--seed N` (default 0), `--ticks N` (260), `--wolves N` (6, also `--ants`).

## Emergence signature (seed 0, 260 ticks, 6 wolves, world 80x44)

The pack closes and PINS the moose: the nearest-wolf distance collapses to ~0.1-0.3
and the largest angular gap shrinks from wide, then oscillates as the pinned pack
jostles.

- seed 0: nearest wolf ~0.3, final gap oscillating ~198/100°.
- seed 1: nearest wolf ~0.1, final gap oscillating ~111/119°.

Numbers differ from the Python reference (which ends ~144-159° / nearest ~0.2) only
because of the different PRNG (SplitMix64 vs. Mersenne Twister gives different wolf
start positions); the pinning/encirclement signature is identical.
