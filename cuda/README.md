# Go to the Ant — CUDA port

A faithful CUDA port of Parunak's foraging swarm ("'Go to the Ant'", Annals of
Operations Research 75:69-101, 1997, §3.1). Behaviour matches the authoritative
Python reference `go_to_the_ant.py`.

## The CUDA lens
The pheromone field **is** GPU global memory, ants **are** threads, and deposits
**are** `atomicAdd` race-resolution — many ants writing the same cell in the same
instant is exactly the stigmergic superposition the paper describes, made literal
by the hardware. Ants parallelize across threads; diffusion and evaporation
parallelize one-cell-per-thread.

## Operationalized deviation (from the sequential Python)
The Python steps ants **sequentially**, so ant *k* sees the fresh deposits of ants
0..k-1 within the same tick. Here ants run in **parallel threads**, so every ant
reads the same pre-tick field snapshot and all deposits land via `atomicAdd`,
becoming visible on the **next** tick — an "all-ants-read-then-write" (Jacobi)
update order. To keep this deterministic, ants read the current food field and
deposit into a fresh copy, and each ant carries its own SplitMix64 stream derived
from `--seed`. The RNG *algorithm* is identical to the other ports; the per-ant
stream partition differs from a single sequential stream (an unavoidable
consequence of parallelism). The emergent signature is unchanged.

## Build

Requires the CUDA toolkit (`nvcc`) and an NVIDIA GPU.

    nvcc -O3 -arch=native -o forage forage.cu

## Run

    ./forage --seed 0                       # defaults: --ticks 3000 --ants 90
    ./forage --seed 1 --ticks 3000 --ants 90 --evap 0.015 --deposit 1.0

Prints the final delivery count, a 20-point S-curve of cumulative deliveries,
and an ASCII render of the food-pheromone field (shades `" .:-=+*#%@"`, nest `N`,
food `F`, obstacle `|`, ants `o`).

## Emergence signature (seed 0)
`deliveries == 0` for the first several hundred ticks (the trail has to form),
then a clear S-curve rise to tens of deliveries by tick 3000 (~48 at seed 0,
~50 at seed 1), with a visible pheromone band connecting nest and food.

---

# Brood sorting — CUDA port

A faithful CUDA port of Parunak's ant brood-sorting swarm ("'Go to the Ant'",
§3.2; Deneubourg et al. 1991). Behaviour matches the authoritative Python
reference `brood_sorting.py`. Grid 40x24, 90 items each of types A/B/C scattered
by a shuffle, 40 ants with short memory (~10), `k+=1 < k-=3`, 120000 ticks.
Clustering = mean fraction of the 8 toroidal neighbours that share an item's type.

## The CUDA lens
The nest grid **is** GPU global memory; ants **are** threads (one per ant). A
pickup or drop is an `atomicCAS` on a grid cell — the ant that wins the race
claims the item (or the empty square), so item conservation is enforced by the
hardware. Sorting is literally many independent threads reshaping one shared
array through local compare-and-swap, with no global controller.

## Operationalized deviation (from the sequential Python)
The Python steps ants **sequentially** on a single shared RNG stream, so ant *k*
acts on the grid already modified by ants 0..k-1 within the same tick. Here the
40 ants run in **parallel threads**, each with its own SplitMix64 stream (derived
from `--seed`), mutating the one shared grid concurrently. `atomicCAS`
(claim-the-cell) keeps items conserved but makes the exact tick-order of
colliding ants depend on GPU scheduling — the grid is **not** bit-identical to
the sequential ports and may vary slightly run to run. The emergent signature is
robust and reproduces. The two probability formulas are PAPER §3.2 VERBATIM.

## Build

    nvcc -O3 -arch=native -o sort sort.cu

## Run

    ./sort --seed 0                         # defaults: --ticks 120000 --ants 40
    ./sort --seed 1 --ticks 120000 --ants 40 --kp 1.0 --km 3.0

Prints ASCII before/after grids (`.` empty, `A`/`B`/`C` items), the initial and
final clustering, and 12 clustering(t) samples across the run.

## Emergence signature (brood sorting)
Initial clustering ~0.26-0.35 (random scatter) rising **monotonically** to
~0.85-0.92 by the end: seed 0 `0.263 -> 0.891`, seed 1 `0.268 -> 0.844`. The
after-grid shows well-separated A / B / C clusters. (Initial value is ~0.26
rather than Python's ~0.35 because the cross-port SplitMix64 shuffle differs from
CPython's PRNG; the rising signature is identical.)

---

# Termites — CUDA port

A faithful CUDA port of Parunak's termite nest-building swarm ("'Go to the Ant'",
Annals of Operations Research 75:69-101, 1997, §3.3; after Kugler et al. 1990).
Behaviour matches the authoritative Python reference `termites.py`.

## The CUDA lens
Both fields — `mass` (persistent structure) and `scent` (decaying pheromone) —
**are** GPU global memory. Termites **are** threads (one per agent). Deposits
**are** `atomicAdd` race-resolution: several termites reinforcing the same growing
column in the same instant is exactly the stigmergic superposition the paper
describes, made literal by the hardware. Evaporation runs one-cell-per-thread.

## Operationalized deviations
- **Formula (from the Python, preserved):** the deposit probability
  `p = min(1, 0.01 + 0.55*(load/maxload) + 0.20*local_scent)`. The paper §3.3
  gives no formula, only "probability rises with local density AND load".
- **Parallel update (language-forced):** the Python steps termites
  **sequentially** (termite *k* sees deposits of 0..*k*-1 within a tick). Here
  termites run in **parallel threads**, all reading the same pre-tick scent
  snapshot and depositing via `atomicAdd` into a fresh copy (Jacobi-style
  "read-then-write"), visible next tick. Against a field that persists across
  40000 ticks (scent decays only 0.02/tick), the one-tick delay leaves the
  column-growing feedback loop unchanged. Each termite carries its own
  SplitMix64 stream from `--seed`; the RNG algorithm matches the other ports but
  the stream partition and parallel deposit order differ from the single
  sequential Python stream, so exact numbers are **not** bit-comparable.

## Build

    nvcc -O3 -arch=native -o termites termites.cu

## Run

    ./termites --seed 0                       # defaults: --ticks 40000 --termites 70 --decay 0.02
    ./termites --seed 1

Prints an ASCII mass render (shades `" .:-=+*#%@"`), the distinct column count and
tallest column mass, and `columns(t)` sampled every `ticks/12` ticks.

## Emergence signature
Scattered dabs self-concentrate into a **handful** of distinct columns (~5-10),
one very tall (tallest mass in the tens of thousands). CUDA seed 0: 5 columns,
tallest ~121192. Seed 1: 5 columns, tallest ~122239. (Exact values differ from
the sequential ports by design — see the parallel-update deviation above.)

---

# Wasp Task Differentiation — CUDA port (`wasps.cu`)

A faithful CUDA port of Parunak §3.4 (Theraulaz et al. 1991). 80 genetically
identical wasps self-separate into **three castes** — one Chief, a band of
Foragers, a Nurse majority — with nobody computing the proportions. Behaviour
matches the authoritative Python reference `wasps.py`.

## The CUDA lens
State lives in GPU global memory (`F[]`, `sig[]`). The parallel rules — the
entropy leak and the foraging/threshold response — run **one thread per wasp**,
and the shared brood-work counter `W` is a literal `atomicAdd` race across all
foragers. But the FACE-OFF spine is irreducibly serial: each duel reads the
freshest force of a **random pair**, transfers a quantum between exactly that
pair, and the next duel must see the result. CUDA makes the split explicit — the
castes emerge in a field of threads, but the dominance ordering that grounds them
is forged one duel at a time.

## Operationalized deviations (marked in source, from the sequential Python)
1. **Face-offs are serialized** in a single-thread kernel (`<<<1,1>>>`) over one
   RNG stream, matching the Python loop order — force conservation between a random
   pair with read-after-write on `F` is a true serial dependency.
2. **Foraging runs in parallel**: each wasp reads the same pre-forage snapshot of
   `D` and `Fmax`, updates its own `sig`, and adds to `W` via `atomicAdd`. Each
   wasp carries its own SplitMix64 stream derived from `--seed`, so the foraging
   draws come from per-wasp streams rather than the single Python stream. The RNG
   *algorithm* is identical to the other ports; the partition differs (parallelism).
   The 3-caste signature is robust and unchanged.

Provenance preserved from the Python: the two Fermi formulas are PAPER VERBATIM;
the entropy-leak force bound (`leak`/`gen`) and `dominance=(F/Fmax)^4` spatiality
proxy are OPERATIONALIZED (see the source header).

## Build

Requires the CUDA toolkit (`nvcc`) and an NVIDIA GPU.

    nvcc -O3 -arch=native -o wasps wasps.cu

## Run

    ./wasps --seed 0                        # defaults: --ticks 4000 --wasps 80
    ./wasps --seed 1 --ticks 4000 --wasps 80

Prints the caste table (n, mean Force, mean Threshold per caste), the Chief's
force vs. population mean, the Forager/Nurse split(t) history, and the
`(Force, threshold)` ASCII landscape.

## Emergence signature (seed 0)
Chief `F~8.9 σ=4.00` (force >> pop mean ~1.25), a small Forager band (~3-6,
`F~5 σ~0`), and a Nurse majority (~74-76, `F~1`) — three castes from identical
wasps. Counts vary slightly with seed and (per the parallel deviation) differ
marginally from the sequential ports; the structure is invariant.

---

# Flocking — CUDA port

A faithful CUDA port of Parunak's flocking system ("'Go to the Ant'", §3.5;
Reynolds 1987 "boids", Heppner 1990). Behaviour matches the authoritative Python
reference `flocking.py`. World 90x48 toroidal, 90 birds, perception radius 8,
separation distance 3, weights sep/ali/coh = 1.3/1.5/0.85, vmax 1.0, turn 0.35,
600 ticks. Reynolds' three local rules (separation, alignment, cohesion), each
normalized to a unit steering vector, sum to turn each bird; a single coherent
flock emerges with no leader.

## The CUDA lens
Every bird's state (px,py,vx,vy) **is** GPU global memory; each bird **is** a
thread. Unlike foraging/brood-sorting, flocking needs **no atomics**: the Python
reference is already a Jacobi update — it computes ALL new velocities from the
pre-tick snapshot and only then moves every bird. That "all-birds-read-old,
all-birds-write-new" structure is exactly the natural GPU formulation, so the
CUDA port is a data-parallel mirror of the sequential reference with no race to
resolve. Each thread does the O(n) neighbour scan; one step is one kernel launch.

## Operationalized deviation (from the sequential Python)
The random init (uniform positions + headings) is done **sequentially on the
host** so the SplitMix64 stream is consumed in exactly the Python order (all px,
then all py, then all headings), keeping init bit-identical to the sequential
ports. The per-tick dynamics carry **no randomness at all** (fully deterministic
given the init), and each thread scans neighbours `j` in ascending index order,
so the neighbour-sum order matches too. No serialized face-off, argmax, or shared
RNG stream exists in this system, so parallelism costs nothing here.

## Build

    nvcc -O3 -arch=native -o flocking flocking.cu

## Run

    ./flocking --seed 0                     # defaults: --ticks 600 --birds 90
    ./flocking --seed 1

Prints an ASCII arrow field (each bird glyph points along its heading), the final
polarization, and a 13-point polarization(t) curve.

## Emergence signature (seed 0)
Polarization (|mean heading|/vmax) starts near 0 (~0.08, disordered) and rises to
~0.9 (one coherent flock): seed 0 `0.08 -> 0.925`, seed 1 `0.14 -> 0.917`. The
arrow field collapses from scattered headings into a single aligned stream.

---

# Wolves — CUDA port

A faithful CUDA port of Parunak's wolf-pack pursuit ("'Go to the Ant'", Annals of
Operations Research 75:69-101, 1997, §3.6; after Korf 1992). Behaviour matches the
authoritative Python reference `wolves.py` and is bit-identical to the sibling
SplitMix64 ports (C, Go, Rust, JS).

## The CUDA lens
The whole world state — the moose and the 6 wolves — lives in GPU global memory.
Each agent's move is an independent argmax/argmin over its 25 candidates (24
directions + stay), so the natural GPU shape is **one thread per agent**: a
single-thread moose kernel (argmax of the min-distance to any wolf) runs first,
then the wolves run as parallel threads, each reading the just-updated moose and
the shared pre-step wolf snapshot. There is no shared field and no write
contention, so the pack is a pure data-parallel map — no atomics needed.

## Operationalized deviations
- **Formula (from the paper, preserved):** each wolf minimises
  `S = d(moose) - k*d(nearest other wolf)` [Korf 1992 VERBATIM]. The continuous
  plane, the 24-candidate direction search (vs the paper's hex grid), and
  `k=1.12`, `vm=0.6`, `vw=1.0` are OPERATIONALIZED.
- **Tiny populations (language-forced, marked):** only 6 wolves and one moose,
  each scanning 25 candidates. The 6-thread wolf kernel and 1-thread moose kernel
  are far below any GPU efficiency threshold — the parallelism is *notional*,
  chosen to express the one-thread-per-agent structure faithfully, not for speed.
- **Update order (bit-faithful, not a divergence):** the Python already uses a
  Jacobi wolf update (every wolf reads the OLD wolf array and the NEW moose, then
  all new positions commit together). The kernels reproduce that order exactly and
  there is **no per-step RNG**, so this port is deterministic and bit-comparable
  with the sequential ports. The init RNG (SplitMix64 from `--seed`) is consumed in
  the same order: per wolf, `uniform(0,w)` then `uniform(0,h)`.

## Build

    nvcc -O3 -arch=native -o wolves wolves.cu

## Run

    ./wolves --seed 0                       # defaults: --ticks 260 --wolves 6
    ./wolves --seed 1

Prints an ASCII board (`M` moose, `W` wolves), the final largest escape gap and
nearest-wolf distance, and `gap deg(t)` sampled every `ticks/12` ticks.

## Emergence signature
The pack closes and **PINS** the moose: the nearest-wolf distance collapses to
~0.1-0.3 while the largest angular gap shrinks from wide and then **oscillates**
(the pinning/encirclement pressure, not a perfectly even ring). CUDA seed 0: final
gap 198 deg, nearest wolf 0.3, gap settling into a 198/100 oscillation. Seed 1:
final gap 111 deg, nearest wolf 0.1, oscillating 111/119. (Identical to the C/Go/
Rust/JS ports; differs from `wolves.py` only because the shared SplitMix64 init
differs from CPython's Mersenne Twister, so the random start positions differ.)
