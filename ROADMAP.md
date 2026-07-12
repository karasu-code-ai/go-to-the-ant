# ROADMAP — CUDA edition

*The GPU-swarm substrate. This branch (`lang/cuda`) owns the CUDA line of "Go to the
Ant." It keeps the shared through-line — stigmergy as coordination through a decaying,
shared medium — and pushes it in the one direction only a GPU can go: **every agent at
once, writing one shared field, with the hardware resolving the race.***

---

## The through-line we keep

Every system in "Go to the Ant" coordinates the same way: **through a shared, decaying
environment, never by direct negotiation.** Ants don't message each other; they modify a
pheromone field and read it back later. The trace an agent leaves *is* the message, and
evaporation keeps the medium honest — stale information fades, so the collective tracks a
moving world without anyone holding global state. Three properties fall out, and they are
why the paradigm survived 25+ years:

1. **The environment is the coordination substrate** — local reads and writes to a field,
   no negotiation.
2. **Decay is a feature** — evaporation is a garbage collector for stale coordination.
3. **Emergence over optimization** — no agent computes the answer; it is a fixed point of
   many cheap local updates plus noise.

The intellectual line runs Parunak 1997 → ACO (Dorigo), PSO (Kennedy & Eberhart), boids
(Reynolds), ABM platforms (Swarm/NetLogo/MASON/Repast), the actor model (Hewitt; Erlang/
Akka) → today's multi-agent RL and **LLM multi-agent systems** (debate, self-consistency,
tool-using swarms). An ant's pheromone field, an ACO graph, a blackboard, and a shared
scratchpad that many LLM samples read and reinforce are **the same object at different
levels of abstraction**: a decaying, shared medium that turns many cheap local
contributions into one global result.

## Why CUDA is the sharpest lens on that object

On a CPU the "shared field" is a metaphor you implement. On a GPU it is the literal
machine:

- The pheromone field **is** global memory.
- Each ant **is** a thread.
- A deposit **is** `atomicAdd` — many ants writing the same cell in the same instant is
  exactly the stigmergic superposition the paper describes, resolved by the hardware.
- A pickup/drop **is** `atomicCAS` — the thread that wins the race claims the item, so
  conservation is enforced by the memory system, not by a lock you wrote.

That is why CUDA is the most direct bridge to the reasoning-model work. **Running a swarm
of reasoning samples on-device is not an analogy for this port; it is the same kernel
shape.** The samples are threads (or blocks), the reinforced-and-decayed partial-solution
structure is the field in global memory, and `atomicAdd` **is** the stigmergic write that
reinforces a shared consensus artifact under contention. See *The distillation bridge*
below.

The honest cost, kept from day one: parallelism reorders the update (all-agents-read-
then-write, a Jacobi step) and partitions the RNG tape per agent, so the discrete deposit
systems do **not** bit-match the sequential ports. They reproduce the *distribution*, not
the trace. That is not a bug to hide — it is rung 5, and it is the whole point of
DETERMINISM.md.

---

## The evolution ladder (CUDA-specific)

Each rung is a real next step a contributor can pick up. They tie to concrete CUDA
features, they are ordered by dependency, and they carry the provenance discipline
through: where the paper prints a formula it stays **VERBATIM**; where it is qualitative
it is **OPERATIONALIZED** and marked in-source.

### Rung 1 — Down the memory hierarchy: shared-memory tiling + warp primitives

The current ports are correct but naive: every deposit and every neighbour scan goes to
global memory. The field ops are stencils and reductions — the two things GPUs reward for
staging in fast memory.

- **Tile the field into `__shared__`.** Diffusion/evaporation (`forage`, `termites`) and
  the flocking neighbour scan are 2-D stencils. Load a halo'd tile into shared memory once
  per block, compute the stencil from there, write back. This is the standard stencil-in-
  shared-memory transform and it makes the field op bandwidth-bound instead of latency-
  bound.
- **Warp-level primitives for the local aggregations.** The per-ant "sum of neighbour
  pheromone" and flocking's "mean heading of neighbours" are reductions. Replace the naive
  loops with `__shfl_down_sync` / `cg::reduce` (cooperative groups) so a warp cooperates on
  one agent's neighbourhood instead of each thread looping alone.
- **Coalesce the field layout.** Store the grid so that threads in a warp touch contiguous
  cells; measure with `ncu` (Nsight Compute) that global loads are coalesced and shared-
  memory access is bank-conflict-free.

*Deliverable:* a `field.cuh` with tiled `diffuse`/`evaporate`/`scan` primitives shared by
`forage`, `termites`, `flocking`, benchmarked against the current global-memory versions
on one GPU. Emergence signatures must be unchanged (same S-curve, same polarization band).

### Rung 2 — The atomics deep-dive (the load-bearing rung)

The deposit is a concurrent read-modify-write on a shared cell — **exactly the contention
pattern of many reasoning samples reinforcing one shared consensus artifact.** This rung
studies it honestly instead of treating `atomicAdd` as a black box.

- **Measure the contention.** When the trail forms, many ants deposit into the same few
  cells — an atomics hot-spot. Profile the serialization; quantify how deposit throughput
  degrades as the field concentrates. This is the stigmergic feedback loop showing up as a
  hardware cost.
- **Contention-reduction strategies, compared honestly:**
  - *Warp-aggregated atomics* — threads in a warp that target the same cell combine their
    contributions with `__match_any_sync` + a warp reduction, then one thread issues a
    single `atomicAdd`. Fewer atomics, same field.
  - *Privatized (per-block) accumulators* in shared memory, flushed to global once per
    block — the classic histogram-privatization pattern, applied to a pheromone grid.
  - *`atomicCAS` retry loops* for the brood-sort claim; measure retry rates as the grid
    sorts and clusters form.
- **Document what each does to determinism.** Warp aggregation changes the summation order
  within a warp; floating-point `atomicAdd` is not associative, so this *changes the
  numbers*. That is the honest link to rung 5 — every contention optimization is a
  reduction-order choice.

*Deliverable:* `notes/atomics.md` with `ncu` numbers, plus a compile-time switch
(`-DDEPOSIT=naive|warpagg|privatized`) so a reader can toggle the strategy and watch both
the throughput and the (distributional) effect on the emergence.

### Rung 3 — One block per colony: thousands of independent swarms

The single most GPU-native scaling move, and the direct enabler for population methods.
Today one run = one colony spread across the grid. Instead: **one CUDA block = one entire
colony**, its field living in that block's shared memory, thousands of blocks running
independent colonies at once.

- **Colony-in-a-block.** For the smaller grids (brood-sort, wasps, wolves, a downscaled
  forage) the whole field fits in shared memory. A block runs its colony end-to-end with
  block-local synchronization (`__syncthreads`, cooperative groups) and never touches
  global memory until it writes its final order-parameter.
- **Massive parameter sweeps for free.** Launch a grid of blocks, each with a different
  `(evap, deposit, k±, radius, seed)`. One kernel launch produces the whole DETERMINISM.md
  Axis-1 sweep — 32 seeds × N parameter settings — instead of a shell `for` loop over
  processes. This turns "sweep the swarm" from an afternoon into a kernel.
- **Population-based methods.** Once colonies are independent and cheap, the natural next
  step is PSO/CMA-ES/island-model *over* colonies: each block evaluates a candidate
  parameter vector, a host (or a second kernel) selects and mutates, the next generation
  launches. This is the concrete on-ramp from "simulate one swarm" to "optimize a
  population of swarms" — and population-of-samples is precisely the distillation frame.

*Deliverable:* `colony_block.cu` running M colonies in M blocks, emitting an
`(params, seed) → order_parameter` table in one launch; a `sweep.cu` that reproduces the
Axis-1 table from DETERMINISM.md as a single kernel.

### Rung 4 — Multi-GPU / NCCL: a field that spans devices

When the field outgrows one GPU (a large ACO graph, a high-resolution grid, or — the real
target — a consensus artifact bigger than one card's memory), the field must be *sharded
across devices* while staying one logical shared medium.

- **Domain decomposition with halo exchange.** Partition the grid across GPUs; each device
  owns a slab and exchanges boundary (halo) cells every tick. This is the same halo
  pattern as rung 1, now across the NVLink/PCIe fabric.
- **NCCL for the halo and for global reductions.** Use `ncclSend`/`ncclRecv` (or
  `ncclAllReduce`) for boundary exchange and for the cross-device order-parameter reduction.
  Overlap the halo exchange with interior compute using separate streams so communication
  hides under computation.
- **Device-initiated communication (stretch).** For fields where deposits cross device
  boundaries irregularly, evaluate NVSHMEM so a thread can write a remote cell directly —
  a *global-address-space* pheromone field. This is the most literal "one shared medium,
  many devices" the stack allows.
- **The reproducibility question sharpens here.** A cross-device reduction has a
  non-fixed order by default. Rung 4 must state, per system, whether the sharded field is
  exact-reproducible (discrete, order-fixable) or distribution-only (chaotic/continuous) —
  the fleet-scale version of DETERMINISM.md's rule.

*Deliverable:* a 2-GPU `forage`/`termites` with NCCL halo exchange; a short
`notes/multigpu.md` recording strong/weak-scaling numbers and the determinism verdict per
system.

### Rung 5 — The honest reckoning with nondeterministic reduction order

This rung is documentation-and-experiment, not new features, and it is where the CUDA
branch earns its keep. Floating-point `atomicAdd` is **not associative**, so the order in
which concurrent deposits land changes the low bits of the field — and the GPU does not
promise an order. This is the parallel-hardware face of the whole reproducibility finding.

- **Quantify the run-to-run jitter.** Run a discrete deposit system (e.g. `sort`,
  `termites`) many times at a *fixed* seed and measure how much the final order-parameter
  moves purely from scheduling nondeterminism. Show it is inside the seed-to-seed spread
  (Axis 1) — i.e. reduction order is a fresh seed for the *micro*-trace, never for the
  *macro* emergence.
- **Offer a deterministic mode.** Provide a build flag that trades speed for exact
  reproducibility: fixed-order deposits via sorted/segmented reduction
  (`cub::DeviceSegmentedReduce`), or a canonical replay of the deposit list, or the
  Kahan/fixed-point accumulation trick for order-independent sums. Show it bit-matches
  across runs (and, for the order-fixable systems, can be made to match the sequential
  ports' distribution far more tightly).
- **Draw the line explicitly.** `wolves` is already bit-identical to C/Go/Rust/JS (Jacobi,
  no atomics, no per-step RNG); `flocking` diverges only through its independent `libm`
  transcendentals under chaos; the atomic systems diverge by *construction*. Keeping those
  three cases distinct in the source and the docs is the deliverable.

*Deliverable:* extend DETERMINISM.md (this branch) with a fixed-seed **run-to-run** jitter
table for the atomic systems, and ship `-DDETERMINISTIC=1` implementing an order-fixed
deposit.

---

## The distillation bridge (public/conceptual)

Rungs 2–5 are, together, the compute pattern of **stigmergic distillation**. Stated at the
same conceptual level as the shared roadmap:

- Treat many independent reasoning samples as a **swarm** — threads or blocks.
- Treat their shared, reinforced-and-decayed intermediate structure as the **pheromone
  field** — a region of global (or multi-device) memory that partial solutions write into.
- Good partial reasoning gets **reinforced and followed** (`atomicAdd` into the shared
  artifact); weak paths **evaporate** (decay applied one-cell-per-thread); a *consensus
  frontier* emerges that no single sample computed — the foraging trail, one level up.

Every CUDA rung maps onto it: shared-memory tiling (rung 1) is how you stage the consensus
artifact in fast memory; the atomics deep-dive (rung 2) is the contention model of many
samples reinforcing one artifact; one-block-per-colony (rung 3) is a population of
independent sample-swarms for sweeps and selection; multi-GPU (rung 4) is a consensus field
too big for one card; and the reduction-order reckoning (rung 5) is exactly the question of
whether a distilled consensus is *reproducible* across a heterogeneous fleet. **`atomicAdd`
is the stigmergic write.** That sentence is the entire bridge.

The security direction rides the same rails: an adversarial swarm that poisons the field or
races the evaporation rate is, on this substrate, a set of threads issuing hostile atomics
into shared memory — and the deterministic/order-fixed mode of rung 5 is one concrete
handle on *what you can guarantee* about a contended shared artifact under attack.

---

## Provenance discipline (carried through every rung)

Where the paper (or its primary source) prints a formula, it is used **VERBATIM** and
tagged in-source: the two Fermi/threshold formulas in `wasps`, the wolf pursuit score
`S = d(moose) − k·d(nearest wolf)` (Korf 1992), the brood-sort pickup/drop probabilities
(§3.2). Where the paper is qualitative, it is **OPERATIONALIZED** and marked: the termite
deposit probability, the wasp entropy-leak and dominance proxy, the continuous-plane
24-direction search. Every CUDA-specific optimization above must preserve these tags —
and any optimization that changes the arithmetic (warp aggregation, fixed-point deposits,
sharded reductions) is documented as a *reduction-order* choice, not a change to the model.
We invent in the gaps, and we say exactly where.

---
*Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
Multi-Agent Systems," Annals of Operations Research 75:69–101 (1997).*
