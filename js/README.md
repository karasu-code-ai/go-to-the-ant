# Go to the Ant — JavaScript ports

Zero-dependency, standard-library-only JavaScript ports of swarm systems from
H. Van Dyke Parunak, *"'Go to the Ant': Engineering Principles from Natural
Multi-Agent Systems,"* Annals of Operations Research 75:69-101, 1997.

- `forage.js` — foraging swarm (§3.1)
- `termites.js` — termite nest building (§3.3)
- `flocking.js` — boids flocking (§3.5)

Runs under Node and is portable to a browser visualizer — the web-native lens.
All 64-bit PRNG arithmetic uses `BigInt` (SplitMix64, identical across every port
so results are directly comparable); the simulation math uses ordinary doubles.

## Run

```sh
node forage.js                 # defaults: 3000 ticks, 90 ants, seed 0
node forage.js --seed 1
node forage.js --ticks 4000 --ants 120 --seed 2
node forage.js --wall          # optional wall-with-a-gap routing demo
```

No build step. Requires Node.js (tested on v20).

## Brood sorting (§3.2)

A second port in this dir: Deneubourg's ant brood sorting (`sort.js`). 40x24
grid, 90 items each of types A/B/C, 40 ants, short memory (~15), pickup
`(k+/(k++f))^2` with k+=0.1, putdown `(f/(k-+f))^2` with k-=0.3, 120000 ticks.

```sh
node sort.js                   # defaults: 120000 ticks, 40 ants, seed 0
node sort.js --seed 1
node sort.js --ticks 60000 --ants 40 --seed 2
```

Flags: `--seed N`, `--ticks N`, `--ants N`.

Output: ASCII before/after grids (`.`=empty, `A`/`B`/`C`=item), a 12-point
`clustering(t)` sample, and the final clustering. Emergence signature: clustering
starts near the random-scatter baseline (~0.26–0.35) and rises monotonically to
~0.85–0.92 as like items coalesce (seed 0 ≈ 0.88, seed 1 ≈ 0.90).

## Flags

`--seed N`, `--ticks N`, `--ants N`, `--evap F`, `--deposit F`, `--wall`.

## Output

The final delivery count, a 20-point S-curve sample of cumulative deliveries,
and an ASCII render of the `food_pher` field (nest `N`, food `F`, ants `o`,
obstacle `|`, pheromone shaded ` .:-=+*#%@`).

## Emergence signature

Deliveries stay 0 for the first few hundred ticks while the trail forms, then
rise on an S-curve to tens of deliveries by tick 3000 (seed 0 ≈ 50, seed 1 ≈ 64;
the trail also diffuses a little — breadth, §3.1/§4.6 — so nearby sub-trails merge),
with a visible pheromone trail connecting nest and food.

---

# Termite nest building — JavaScript port (§3.3)

Termite mound building (Kugler/Turvey). Two fields: `mass` (persistent
structure) and `scent` (decaying pheromone that biases wandering). Scattered
dabs self-concentrate into a handful of tall columns — no termite plans the
mound.

## Run

```sh
node termites.js                 # defaults: 40000 ticks, 70 termites, seed 0
node termites.js --seed 1
node termites.js --ticks 40000 --termites 70 --decay 0.02 --seed 2
```

No build step. Requires Node.js (tested on v20).

## Flags

`--seed N`, `--ticks N`, `--termites N` (alias `--ants`), `--decay F`.

## Output

An ASCII render of the `mass` field (shaded ` .:-=+*#%@`), the distinct-column
count, the tallest column mass, and a 13-point `columns(t)` trace.

## Emergence signature

Scattered dabs self-concentrate into a HANDFUL of distinct columns (~5–10), one
very tall (tallest mass in the tens of thousands). Seed 0: 5 columns, tallest
≈ 103360; seed 1: 4 columns, tallest ≈ 56810. These numbers are bit-identical
to the other sequential ports (Go/Java/Rust/C) via the shared SplitMix64
convention; they intentionally differ from the CPython reference, which uses a
different base PRNG.

---

# Boids flocking — JavaScript port (§3.5)

Reynolds' three local rules — separation, alignment, cohesion — on a toroidal
90x48 world with 90 birds. Each urge is normalized to a unit vector before the
weighted sum steers the bird. Fully deterministic given the random init (no
per-step randomness). A single coherent, banking flock emerges with no leader.

The perception radius (`perc=8`), separation distance (`sep_r=3`), and the three
weights (`wsep=1.3`, `wali=1.5`, `wcoh=0.85`) are OPERATIONALIZED — Parunak lists
the rules but gives no numbers (Reynolds 1987 is the primary source for tuned
constants).

## Run

```sh
node flocking.js                 # defaults: 600 ticks, 90 birds, seed 0
node flocking.js --seed 1
node flocking.js --ticks 800 --birds 120 --seed 2
```

No build step. Requires Node.js (tested on v20).

## Flags

`--seed N`, `--ticks N`, `--birds N` (alias `--ants`).

## Output

An ASCII arrow field (each bird points along its heading), the final
polarization, and a 13-point `polarization(t)` trace.

## Emergence signature

Polarization (`|mean heading|/vmax`) starts near 0 (disordered scatter, ~0.03–
0.15) and rises to ~0.9 as the birds converge into one coherent flock. Seed 0:
0.08 → 0.84; seed 1: 0.14 → 0.89. (These numbers use the shared SplitMix64
convention and intentionally differ from the CPython reference, whose
`random.Random` init gives seed 0: 0.08 → 0.917.)

---

# Wasp task differentiation — JavaScript port (§3.4)

Theraulaz et al. (1991) wasp caste differentiation. `n=80` genetically IDENTICAL
wasps, 4000 ticks. Three interacting rules — face-offs that pass a quantum of
Force (Fermi win probability `1/(1+e^(h(Fi-Fj)))`, VERBATIM), brood demand
`D+=appetite-W`, and foraging that lowers/raises a threshold (Fermi
`1/(1+e^(hf(sig-D)))`, VERBATIM). Two mechanisms are OPERATIONALIZED and tagged
in-source: the force-relaxation bound (`leak`/`gen` replacing an ad-hoc cap — an
inference beyond Parunak, NOT the §4.6 entropy leak, which is Rule 1's force flow)
and the LOCAL `dominance=(F/seenmax)^4` spatiality proxy — each wasp's own fading
memory of the top force it has faced (no global max) — that restores the Chief's high threshold.

## Run

```sh
node wasps.js                  # defaults: 4000 ticks, 80 wasps, seed 0
node wasps.js --seed 1
node wasps.js --ticks 4000 --wasps 80 --seed 2
```

No build step. Requires Node.js (tested on v20).

## Flags

`--seed N`, `--ticks N`, `--wasps N` (alias `--ants`).

## Output

A caste table (n, mean Force, mean Threshold per caste), the Chief's force vs the
population mean, a 13-point `Forager/Nurse split(t)` trace, and the `(F,sig)`
ASCII landscape (`C` chief, `F` forager, `n` nurse; x = Force, y = Threshold).

## Emergence signature

From genetically-identical wasps, THREE castes emerge: exactly 1 Chief (high
force, HIGH threshold ~4), a small band of Foragers (~2–6, force ~5, threshold
~0), and a Nurse majority (~70+, force ~1). Chief force >> population mean. Seed
0: Chief F=9.78 σ=4.00, 5 Foragers F≈5.15, 74 Nurses F≈0.88; seed
1: Chief F=9.59, 4 Foragers, 75 Nurses. These numbers are bit-identical to the
other sequential ports (Go/Java/Rust/C) via the shared SplitMix64 convention;
they intentionally differ from the CPython reference, which uses a different base
PRNG (the STRUCTURE — the three-caste split — is what is preserved).

---

# Wolves surrounding prey — JavaScript port (§3.6)

Korf's (1992) pack pursuit. 6 wolves, an 80x44 continuous plane, 260 ticks. The
moose starts at centre; wolves start uniform-random. Two local rules, no
communication: the MOOSE moves to the candidate FARTHEST from its nearest wolf;
each WOLF moves to minimise `S = d(moose) - k*d(nearest other wolf)` (VERBATIM
score) — close on the prey while spreading out from the other wolves. Each agent
searches 24 candidate directions (`x+sp*cos(a), y+sp*sin(a)`) plus staying put.
Fully deterministic given the random init (no per-step randomness).

The continuous plane + 24-candidate search (replacing the paper's hex grid), the
speeds (`vm=0.6`, `vw=1.0`), and `k=1.12` are OPERATIONALIZED and tagged
in-source; the two rules and `S = d(moose) - k*d(wolf)` are VERBATIM.

## Run

```sh
node wolves.js                 # defaults: 260 ticks, 6 wolves, seed 0
node wolves.js --seed 1
node wolves.js --ticks 400 --wolves 6 --seed 2
```

No build step. Requires Node.js (tested on v20).

## Flags

`--seed N`, `--ticks N`, `--wolves N` (alias `--ants`).

## Output

An ASCII board (`M` moose, `W` wolves), the final largest escape gap and
nearest-wolf distance, and a 14-point `gap°(t)` trace.

## Emergence signature

The pack closes and PINS the moose: the nearest-wolf distance collapses to ~0.1–
0.3, and the largest angular gap starts wide, shrinks as the ring forms, then
oscillates as the wolves jockey against the fleeing moose. Seed 0: nearest wolf
0.3, gap oscillating ~100–198°; seed 1: nearest wolf 0.1, gap oscillating ~111–
119°. These numbers use the shared SplitMix64 convention and are bit-identical to
the other sequential ports (Go/Java/Rust/C); they intentionally differ from the
CPython reference, whose `random.Random` init gives different start positions
(the emergence — encirclement PRESSURE and pinning, not a perfectly even ring —
is what is preserved).
