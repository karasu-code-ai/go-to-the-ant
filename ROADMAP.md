# Go to the Ant — the **C** edition roadmap: bare metal & the substrate

This is the C branch's own forward line. It keeps the shared through-line — stigmergy as coordination
through a decaying shared medium — but commits to the question only C can answer honestly: **how cheap can a
coordinating agent physically get, and what is the field when nothing is hidden?**

## The through-line: stigmergy (kept)

Every system here coordinates the same way — **through a shared, decaying environment**, never by direct
negotiation. Ants don't message each other; they modify a pheromone field and read it back later. Parunak's
word (from Grassé) is **stigmergy**: the trace an agent leaves in the world *is* the message, and evaporation
keeps the medium honest — stale information fades, so the collective tracks a moving world without anyone
holding global state. Three properties fall out and are the whole reason the paradigm survived 25+ years:

1. **The environment is the coordination substrate** — local reads and writes to a field, no lock, no broadcast.
2. **Decay is a feature** — evaporation is a garbage collector for stale coordination.
3. **Emergence over optimization** — no agent computes the answer; it is a fixed point of many cheap local
   updates plus noise.

The straight line the whole project draws: an ant's pheromone field, an ACO graph, a blackboard, and a shared
scratchpad that many LLM samples read and write are **the same object at different levels of abstraction** — a
decaying, shared medium that turns many cheap local contributions into one global result. C's job is to be the
level where that object is *literally a block of memory* and you can count the cost of touching it.

## The C lens: the field is memory

In every other port the field is dressed up — an object, a typed array behind a runtime, a device buffer. In C
it is what it actually is: `double food_pher[H*W]`, flat index arithmetic, a hand-rolled SplitMix64, manual
`malloc`/`free`, `-lm` for the transcendentals. Deposit is a store. Evaporate is a scaled load-store sweep.
Follow-the-gradient is a stencil read. There is **no abstraction between agent and field** — which is exactly
why C is the reference the other five ports are measured against, and why the DETERMINISM finding can say "C
and Rust are bit-identical because they link the same system `libm`." C is the ground truth.

That gives C a distinct mandate the other branches don't have: **own the floor on per-agent cost, and be the
ABI everyone else calls into.** The rungs below are all in service of those two things.

## The evolution ladder (C-specific rungs)

Each rung is a real next step a contributor could pick up, tied to concrete C features and tooling — not vibes.

### Rung 1 — Shrink it: the arena, no `malloc` in the hot loop

The current ports allocate their fields once, which is fine, but the agents and scratch state are still ad hoc.
Make the memory discipline explicit and measurable:

- **Fixed-size arenas / bump allocators.** One up-front `mmap` (or a static buffer) for the field, one for the
  agent pool. Zero allocation inside the tick loop; freeing is resetting a pointer. This is the pattern that
  makes the per-agent cost a *known constant* rather than an allocator's whim.
- **Struct-of-arrays, not array-of-structs.** The flocking port already hints at this (`px/py/vx/vy` as four
  flat lanes). Push it everywhere: SoA is what lets the evaporate/diffuse sweeps stream linearly through cache
  and what makes rung 2 (SIMD) possible at all. AoS `struct ant { ... }` is the thing to delete.
- **Cache-aware field layout.** Blocked/tiled traversal of the 2-D field so a diffusion stencil reuses rows
  already in L1; consider a padded stride to avoid false sharing once rung 4 threads the sweep. Measure with
  `perf stat` (cache-misses, IPC) and keep the numbers in the README, the same way emergence signatures are
  kept now.
- **Deliverable:** an `arena.h` single-header used by all six systems, plus a `bench` target that reports
  ns/agent/tick and cache-miss rate at a fixed seed. This *is* the "how cheap can a coordinating agent get"
  measurement, made reproducible.

### Rung 2 — SIMD the field ops (evaporate / diffuse / gradient)

The field-wide operations — evaporation (`f *= 1-evap`), diffusion (a neighbour-average stencil), and the
force/scent decay in termites and wasps — are embarrassingly data-parallel and dominate the tick for large
grids. This is where C earns its keep:

- **Intrinsics first, portably.** `<immintrin.h>` (AVX2/AVX-512) on x86, `<arm_neon.h>` on the ARM targets
  rung 3 cares about. Gate them behind a tiny `simd.h` with a scalar fallback so the port still builds
  `-std=c11` everywhere and the DETERMINISM guarantees hold on the scalar path.
- **Watch the arithmetic contract.** Evaporate/diffuse are `+ - * /` only, so a well-ordered SIMD sweep stays
  bit-identical to the scalar reference — but horizontal reductions and FMA (`-ffp-contract`) can change the
  last ULP. Keep FMA **off** on the reference build, or add a `--fast-math` variant that is explicitly marked
  as *distribution-reproducible only* (see the DETERMINISM banner). This is a place C can make the
  reproducibility boundary an actual compile flag.
- **Deliverable:** vectorised `evaporate()` and `diffuse()` with a scalar-vs-SIMD bit-equality test in CI, and
  a speedup table. The stencil work here is the direct on-ramp to rung-adjacent GPU kernels — the SIMD lane and
  the CUDA thread are computing the same field update.

### Rung 3 — Embedded / real-time swarms: stigmergy where compute is scarce

This is C's *unique* rung — the one no other branch can take. Push a port down onto a microcontroller or sensor
mote and prove that coordination survives with kilobytes of RAM and no OS:

- **Fixed-point / `int16` fields.** Drop `double` for a scaled integer field where an FPU is absent or
  expensive; evaporation becomes a shift-and-subtract. Quantify the emergence loss (does brood-sort still reach
  ~0.9 clustering?) — a genuinely new result, not a reimplementation.
- **No heap, static footprint, bounded tick time.** Everything from rung 1's arena but sized for
  `-mcpu=cortex-m*`: link with `-nostdlib` where possible, replace `-lm` with a small polynomial `sin`/`cos`
  (flocking/wolves need it), and report worst-case tick latency so it fits a real-time budget.
- **Field-as-shared-memory across motes.** A handful of MCUs writing a shared low-power radio / shared-SRAM
  region *is* a physical pheromone field — the trace in the medium is the message, exactly the paper's model,
  now with real energy and bandwidth costs. Stigmergy is the *right* paradigm when compute is scarce precisely
  because it needs no consensus protocol.
- **Deliverable:** one system (brood-sort or foraging is the natural first) building for a Cortex-M target with
  a static RAM budget in the README, plus the fixed-point-vs-double emergence comparison.

### Rung 4 — Generalize the field: the `field_t` interface behind one ABI

Now do the shared ladder's "generalize the field" rung the C way — as a **stable C ABI**, because C is where
the boundary is naturally an ABI and every other language and the GPU already know how to call C:

- **A minimal vtable-free interface:** `field_deposit`, `field_read`, `field_evaporate`, `field_diffuse` over
  an opaque `field_t*`, with the concrete layout (dense `double`, fixed-point, tiled) chosen at construction.
  The *agent* and the *field* become swappable — the refactor that later lets an "agent" become a reasoning
  sample and a "pheromone" become a reinforced partial solution.
- **`extern "C"` reference for the polyglot.** Export it as a shared object with a clean header so Rust
  (`bindgen`/FFI), Go (`cgo`), JS (WASM/N-API), and the CUDA host can link the *same* field implementation.
  This makes C's field the literal reference the DETERMINISM cross-check runs against, not just a sibling.
- **Deliverable:** `libfield.so` + `field.h`, one system rebuilt on top of it, and a smoke test that a second
  language links and reproduces C's numbers bit-for-bit at a fixed seed.

### Rung 5 — The campaign bridge: prove the per-agent floor, be the substrate

The two campaign directions the whole project points at are **stigmergic distillation** (many reasoning samples
as a swarm; their reinforced-and-decayed shared structure as the pheromone field / consensus frontier) and
**polyagentic security** (adversarial vs defensive swarms coordinating through a shared environment). C's
contribution to both is specific and unglamorous, which is the point:

- **The field-as-shared-memory is exactly the substrate a GPU / distributed distillation swarm sits on.** A
  pheromone grid under concurrent atomic deposit *is* the compute pattern of running a swarm of samples on a
  device; the C `field_t` (rung 4) is the host-side reference that the CUDA branch's `atomicAdd` field is
  validated against. C proves the sequential semantics the parallel version must reproduce *in distribution*.
- **C is where you prove the per-agent cost.** Rungs 1–3 turn "how cheap can a coordinating agent get" from a
  slogan into a number: ns/agent/tick, bytes/agent, worst-case latency on bare metal. That floor is what tells
  you whether a stigmergic layer is affordable underneath a fleet of expensive reasoning agents — the coordination
  substrate should cost approximately nothing next to the agents it coordinates, and C is the branch that can
  show it does.
- **For security:** the raw-memory field makes the "who is allowed to write the field" question physical —
  bounds, ownership of a byte range, what a forged deposit costs. Where Rust *proves* it with the type system,
  C *measures* it and defines the ABI an attacker or defender actually operates on.

Provenance discipline carries through every rung: where the paper (or its primary sources) prints a formula it
is used **verbatim and tagged** (the foraging/brood pickup-putdown probabilities, both wasp Fermi rules, the
wolf `S = d(moose) - k·d(nearest wolf)` score); where it is qualitative it is **operationalized and marked**
(the termite deposit probability, the flocking radius/weights, the wolf continuous-plane serialization). We
invent in the gaps — and say exactly where.

---
*Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural Multi-Agent Systems,"
Annals of Operations Research 75:69–101 (1997).*
