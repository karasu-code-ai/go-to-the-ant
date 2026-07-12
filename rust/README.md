# Go to the Ant — Rust ports

Faithful single-file ports of Parunak's swarm systems (Parunak 1997),
`std`-only, no external crates.

## Flocking (`flocking.rs`, §3.5)

**Rust lens:** ownership and borrowing make *"who is allowed to mutate the shared
state this step"* explicit and machine-checked. The whole flock (positions +
velocities) lives in one `Flock`; `step` computes every bird's NEXT velocity into
freshly-owned `nvx/nvy` buffers while holding only shared (`&`) reads of the
current state, then commits them in a second pass — so the double-buffered "every
bird sees the same snapshot this tick" semantics are not a convention you must
remember but a fact the borrow checker enforces: no bird can mutate the field
another bird is still reading.

Boids: n=90, world 90x48, perception 8, separation 3, weights sep 1.3 / align 1.5
/ cohesion 0.85, vmax 1.0, turn 0.35, 600 ticks. Reynolds' three local rules
(separation, alignment, cohesion) are the paper's; the radii and weights are
OPERATIONALIZED (Reynolds 1987 supplies the tuned constants Parunak omits). NO
per-step randomness — fully deterministic given the random init.

### Build / Run

```
rustc -O flocking.rs
./flocking --seed 0
```

Flags: `--seed N`, `--ticks N` (default 600), `--birds N` / `--ants N` (default 90).
Output: the ASCII arrow field (each bird an arrow along its heading), the final
polarization, and a 13-point `polarization(t)` series.

### Emergence signature (seed 0)

Polarization (|mean heading| / vmax) starts near 0 and rises to one coherent
flock: seed 0 goes 0.08 → 0.898, seed 1 goes 0.14 → 0.874. PRNG is SplitMix64
(shared across all ports), so the exact init differs from the Python/Mersenne-
Twister reference (seed 0: 0.08 → 0.917) but the same near-0 → ~0.9 rise holds.

---

## Foraging (`forage.rs`, §3.1)

**Rust lens:** ownership answers *"who owns the shared field?"*. The two pheromone
fields live in one `World`; ants borrow it mutably one at a time in the tick loop,
so the "communication through the environment" channel is a single-owner resource
mutated in strictly serial order — the borrow checker machine-checks that no two
ants alias the field at once, making the Python's implicit serial semantics explicit.

## Wasps (`wasps.rs`, §3.4)

**Rust lens:** ownership and borrowing make *"who is allowed to mutate the shared
state this step"* explicit and machine-checked. The colony's force/threshold
vectors live in one `Colony`; each of the three coupled rules (face-offs,
entropy-leak, foraging/demand) borrows `&mut self` for the whole step, so they run
in a strictly serial, single-owner order the borrow checker guarantees — exactly
the Python semantics, made explicit.

Build/run:

```
rustc -O wasps.rs
./wasps --seed 0
```

Flags: `--seed N`, `--ticks N` (default 4000), `--wasps N` (default 80; `--ants`
also accepted). Output: the caste table (n, mean Force, mean Threshold per caste),
the `(Force, Threshold)` ASCII landscape, and the Forager/Nurse split(t) series.

**Emergence signature (seed 0):** from 80 genetically identical wasps, THREE castes
self-separate — exactly 1 Chief (Force 9.78, high Threshold 4.00), a small band of
4 Foragers (Force ~4.9, Threshold ~0), and a Nurse majority of 75 (Force ~0.95).
Chief force >> pop mean (1.25). Seed 1: Chief 9.59, 3 Foragers ~5.5, 76 Nurses.

## Wolves (`wolves.rs`, §3.6)

**Rust lens:** ownership and borrowing make *"who is allowed to mutate the shared
state this step"* explicit and machine-checked. Each step is two phases: the moose
reads all wolves and moves, then every wolf reads the moose and all OTHER wolves and
moves. The wolf phase borrows the OLD `wolves` slice immutably and writes into a fresh
`next` Vec that replaces the pack only after the whole phase — so no wolf ever sees a
half-updated pack, making the Python's implicit simultaneous-wolf-update explicit and
alias-free.

6 wolves, world 80x44, moose speed vm=0.6, wolf speed vw=1.0, k=1.12, 260 ticks. The
moose starts at centre; wolves uniform-random. Each agent searches 24 directional
candidates plus staying put; the moose maximises min-distance to any wolf, each wolf
minimises `S = d(moose) - k*d(nearest other wolf)` (VERBATIM). Continuous plane +
24-candidate search replace the paper's hex grid (OPERATIONALIZED). No per-step
randomness — deterministic given the random init.

Build/run:

```
rustc -O wolves.rs
./wolves --seed 0
```

Flags: `--seed N`, `--ticks N` (default 260), `--wolves N` (default 6; `--ants` also
accepted). Output: the ASCII `M`/`W` board, the final largest escape gap and
nearest-wolf distance, and a 14-point `gap(t)` series.

**Emergence signature (seed 0):** the pack closes and PINS the moose — nearest-wolf
distance drops to ~0.3 while the largest angular gap shrinks from wide (161) and then
oscillates (198/100). Seed 1: nearest wolf 0.1, gap settles into an oscillation
(111/119). The emergence is the pinning/encirclement pressure, not a perfectly even
ring. PRNG is SplitMix64 (shared across all ports), so exact gap numbers differ from
the Python/Mersenne-Twister reference (seed 0: nearest 0.2, gap 159/144) but the same
pinning-and-oscillation signature holds.

## Build

```
rustc -O forage.rs
```

## Run

```
./forage --seed 0
```

Flags: `--seed N`, `--ticks N` (default 3000), `--ants N` (default 90),
`--evap F` (default 0.015), `--wall` (optional wall-with-gap routing demo).

Output: the ASCII trail render (nest `N`, food `F`, obstacle `|`, ants `o`,
pheromone density ` .:-=+*#%@`), the final delivery count, and a 20-point
cumulative-delivery S-curve sample.

## Emergence signature (seed 0)

Deliveries stay 0 for the first several hundred ticks (the trail must form first),
then rise on an S-curve to 46 by tick 3000. Seed 1 gives 60. A pheromone trail
connecting nest and food is visible in the render. PRNG is SplitMix64 (shared
across all ports), so exact numbers differ from the Python/Mersenne-Twister
reference but the qualitative signature matches.

---

# Brood sorting (§3.2) — `sort.rs`

Faithful single-file port of Deneubourg's ant brood sorting (Deneubourg et al.
1991), `std`-only. Grid 40x24 seeded with 90 items each of types A/B/C (270
total), 40 ants with a 10-step memory, `k+`=1 < `k-`=3, 120000 ticks. Ants
wander, remember recently-seen types, pick up rare-locally items
(`p=(k+/(k+ + f))^2`) and drop them among like items (`p=(f/(k- + f))^2`). Sorted
clusters EMERGE with no ant comparing the whole nest.

**Rust lens:** ownership and borrowing make *"who is allowed to mutate the shared
grid this step"* explicit and machine-checked — each ant takes a `&mut` borrow of
the single-owner `Nest` one at a time in the tick loop, so the pick-up/put-down
that edits shared state is a serial mutation the borrow checker proves is
alias-free.

## Build

```
rustc -O sort.rs
```

## Run

```
./sort --seed 0
```

Flags: `--seed N`, `--ticks N` (default 120000), `--ants N` (default 40).

Output: ASCII before/after grids, the initial and final clustering, and a
12-point `clustering(t)` sample.

## Emergence signature

Clustering (mean fraction of the 8 toroidal neighbours sharing an item's type)
starts scattered and rises monotonically to a well-sorted field: seed 0 goes
0.26 → 0.876, seed 1 goes 0.27 → 0.903. (SplitMix64 differs from the Python
Mersenne-Twister, so the initial scatter value differs from the reference's 0.35,
but the same rise-to-~0.85-0.92 signature holds.)

---

## Termites (`termites.rs`, §3.3)

Faithful single-file port of Parunak's termite nest-building system (Kugler et
al. 1990), `std`-only, no external crates.

**Rust lens:** ownership makes *"who is allowed to mutate the shared state this
step"* explicit and machine-checked. The two fields (`mass`, `scent`) live in one
`Mound`; termites borrow it mutably one at a time in the tick loop, so the
stigmergic channel is a single-owner resource mutated in strictly serial order —
the borrow checker guarantees no two termites alias the field at once.

### Build

```
rustc -O termites.rs
```

### Run

```
./termites --seed 0
```

Flags: `--seed N`, `--ticks N` (default 40000), `--termites N` / `--ants N`
(default 70), `--decay F` (default 0.02).

Output: the ASCII mass-density render (columns emerge as bright cores, shades
` .:-=+*#%@`), the final distinct-column count and tallest-column mass, and a
13-point `columns(t)` trajectory.

### Emergence signature (seed 0)

Scattered dabs self-concentrate into a HANDFUL of distinct columns, one very
tall. `columns(t)` climbs to ~14 as dabs scatter, then consolidates back down to
a handful: seed 0 → 5 columns, tallest mass ~96053; seed 1 → 4 columns, tallest
~133850. PRNG is SplitMix64 (shared across all ports), so exact numbers differ
from the Python/Mersenne-Twister reference (seed 0: 7 columns, ~92659) but the
qualitative signature — a few tall columns condensing out of noise — matches.
