# Go to the Ant — Java edition roadmap

*The ABM-platform lineage, and the JVM's path to large-scale instrumented swarms.*

This is the Java line of a six-language recreation of Parunak's foraging swarm. The shared story
is intact and lives below; this document is the part that is **ours** — where the Java ports go next,
tied to real JVM features a contributor can pick up today.

## The through-line we keep: stigmergy

Every system in "Go to the Ant" coordinates the same way — **through a shared, decaying environment**,
never by direct negotiation. Ants don't message each other; they modify a pheromone field and read it
back later. Parunak's word for it (from Grassé) is **stigmergy**: the trace an agent leaves in the
world *is* the message, and evaporation keeps the medium honest — stale information fades, so the
collective tracks a moving world without anyone holding global state. Three properties follow, and
they are why the paradigm survived 25+ years:

1. **The environment is the coordination substrate** — local reads/writes to a field, no lock, no broadcast.
2. **Decay is a feature** — evaporation is a garbage collector for stale coordination.
3. **Emergence over optimization** — no agent computes the answer; it is a fixed point of many cheap local updates plus noise.

The intellectual line runs Parunak 1997 → **ACO** (pheromone-on-a-graph), **PSO** and **boids**
(the flocking rules as optimizers and crowd control), **ABM platforms**, the **actor model**, and
finally **LLM multi-agent systems**. An ant's pheromone field, an ACO graph, a blackboard, and a
shared scratchpad that many LLM samples read and write are **the same object at different levels of
abstraction** — a decaying, shared medium that turns many cheap local contributions into one global result.

## Why Java holds this lens

The ABM-platform lineage *is* Java's lineage. **Swarm** begat **Repast**, **MASON**, and (on the JVM)
**NetLogo** — the tools that made "many local agents in a shared field" a first-class scientific
instrument rather than a one-off simulation. When a computational biologist or economist reaches for a
serious agent-based model, they reach for something JVM-shaped. That is not incidental to this port; it
is the whole reason Java is one of the lenses. The other five ports ask *what is the field* (C), *what
if every agent runs at once* (CUDA), *who may write it* (Rust), *what if each agent is a process* (Go),
*who gets to see it* (JS). **Java asks: how did the field's scientists actually build the instrument —
and how far does it scale?**

Today the six Java ports already lean into the idiom: a `World`/`Nest`/`Mound`/`Colony`/`Hunt` object
owns the shared field(s), and an explicit per-tick schedule steps a swarm of individually-instantiated
agent objects (`Ant`, `SortAnt`, `Termite`, `Wasp`, `Boid`, wolves). That is a *sketch* of the
platform architecture. The roadmap below turns the sketch into the real thing and then pushes it to a
scale the other lenses can't reach.

## The Java evolution ladder

Each rung is a concrete, pickable piece of work tied to a specific JVM capability. Rungs build on each other.

### Rung 1 — Extract the MASON/Repast schedule abstraction

Right now each of the six ports hand-rolls its own tick loop and its own field. Factor out the two
primitives every ABM platform is built on:

- **`Steppable`** — a one-method interface, `void step(Schedule s)`, that every agent (and the
  field-evaporation step) implements. This is MASON's exact contract; adopting the name is deliberate,
  so the ports read as real ABM code.
- **`Schedule`** — an ordered driver that steps registered `Steppable`s per tick, with explicit
  *ordering epochs* so we can preserve the semantics the reference depends on: agents-read-then-field-evaporates
  (foraging, termites), or two-phase compute-then-commit (flocking, wolves). The ordering discipline
  that makes the ports deterministic (see below) becomes a first-class, inspectable property of the
  `Schedule` rather than an implicit loop.
- **`Field2D` / `ScalarField`** — the shared medium as an interface: `read(x,y)`, `deposit(x,y,amt)`,
  `evaporate(rate)`, toroidal wrap. C's raw array is one implementation; a sparse/tiled field is another.
  Repast/MASON call this a `Grid` / `ObjectGrid2D`; matching the vocabulary is the point.

The payoff is exactly the shared ladder's rung 3 ("generalize the field"), realized in Java's idiom:
once `Steppable` and `Field2D` are interfaces, **the agent and the field become swappable** — the
precondition for a "pheromone" becoming a reinforced partial solution and an "ant" becoming a reasoning
sample. Deliverable: all six ports rebuilt on one `swarm.core` package; behaviour bit-identical to today
(the cross-language check is the regression test).

### Rung 2 — Project Loom: agent-per-virtual-thread

This is the rung only Java can climb this way. The actor-model cousin of stigmergy says *an agent is a
process that reacts to its local neighbourhood*. Go expresses that with goroutines; the JVM's answer,
GA since Java 21, is **virtual threads**. Give every agent its own virtual thread:

- Each agent is a loop: read local field patch → apply rules → deposit → await the tick barrier. Millions
  of virtual threads are cheap (a few hundred bytes of heap each, parked on a carrier thread when blocked),
  so a swarm of 10^5–10^6 ants stops being a `for` loop over an array and becomes a genuine population of
  concurrent agents.
- Coordination stays **stigmergic**, not message-passing: agents never call each other; they contend on
  the shared `Field2D`. That makes the write-conflict question real — a `deposit` is a concurrent
  read-modify-write, the JVM analogue of CUDA's `atomicAdd`. Options to implement and compare:
  `LongAdder`/`DoubleAdder` per cell, `AtomicLongArray` with fixed-point accumulation, or a
  `StructuredTaskScope` per tick that fans out all agents and joins at the barrier before the field
  evaporates.
- This directly instantiates the **all-read-then-write vs. read-modify-write** modelling choice that
  CUDA surfaced — now as a scheduling decision inside one JVM, where you can toggle it and measure the
  behavioural delta on the same machine.

Honest caveat, tagged as such: concurrent floating-point deposits are **not** order-deterministic
(`double` addition isn't associative), so a parallel Java run reproduces the *distribution*, not the
exact trace — the same boundary DETERMINISM.md draws for CUDA. Fixed-point (`long`) deposits restore
associativity and let a parallel run stay bit-exact; documenting that trade-off is part of the deliverable.

### Rung 3 — JMH: make throughput a measured result, not a vibe

"Scales to many agents" is a claim; on the JVM you settle claims with **JMH** (the Java Microbenchmark
Harness), which handles warmup, JIT steady-state, and dead-code elimination that naive timing gets wrong.
Build a `swarm-bench` module that reports, with proper error bars:

- ticks/second and agent-steps/second vs. population size (10^3 → 10^6) for the sequential loop, the
  virtual-thread version, and a plain platform-thread pool — the three side by side.
- field-update throughput under contention for each `deposit` strategy from Rung 2.
- allocation and GC pressure per tick (agents as objects vs. struct-of-arrays), with an eye toward
  **Project Panama / Vector API** and (later) **Valhalla value objects** to shrink per-agent footprint.

The result is a throughput/scaling curve that turns "the JVM is the platform for large ABM" from lineage
into a number. This is also what makes a swarm run publishable as a *reproducible performance experiment*.

### Rung 4 — Interop with the JVM data/ML ecosystem

The reason to run a swarm *on the JVM specifically* is everything already living there. Wire the ports into it:

- **Instrumentation & provenance:** emit per-tick order parameters (deliveries, clustering, column count,
  polarization, Chief force) as structured records; sink to Parquet/Arrow so a run becomes a queryable
  dataset, not console text. Tag every metric with its provenance (`VERBATIM` vs `OPERATIONALIZED`) in the
  schema, carrying the discipline into the data layer.
- **Analysis:** feed those datasets to the JVM data stack (Tablesaw / Arrow, or out to Spark for
  seed-sweeps at fleet scale) so an across-seed distribution study is a query, not a shell loop.
- **Reasoning agents (the campaign bridge):** the `Steppable` contract is deliberately model-agnostic. A
  `Steppable` whose `step` calls a model runtime on the JVM (ONNX Runtime, DJL, or a remote inference
  service) is a *reasoning* agent dropped into the same stigmergic scaffold — the ant replaced by a
  sample, the pheromone replaced by a reinforced partial solution, the `Schedule` and `Field2D` unchanged.

## The bridge to the campaign

The shared work has two directions at the end of the line, and the Java ladder is a specific on-ramp to both:

- **Stigmergic distillation.** Treat many independent reasoning samples as a swarm and their shared,
  reinforced-and-decayed intermediate structure as the pheromone field: good partial reasoning is
  reinforced and followed, weak paths evaporate, a *consensus frontier* emerges that no single sample
  computed — the foraging trail, one level up. Java's contribution is the **platform view**: once agents
  are `Steppable`s on virtual threads over a shared `Field2D`, swapping the random-walk ant for a
  reasoning sample (Rung 4) is a change of implementation, not architecture — and Rung 3 tells you whether
  a population of thousands of samples is throughput-feasible on the hardware you have. This is the path
  to a distillation swarm run as a **reproducible scientific experiment** with the full ABM/ML toolchain
  around it: scheduled, instrumented, logged to Arrow, swept across seeds, analyzed like any other
  large-scale simulation.

- **Polyagentic security.** A swarm is a threat model *and* a defense: adversarial agents that coordinate
  through the shared field (poisoning it, exploiting the evaporation rate, forging trails) versus defensive
  swarms that detect and re-route. On the JVM the `Field2D` write path is exactly where "who is allowed to
  deposit, and how much" gets enforced — an access question the platform layer can instrument and audit at
  scale, complementing Rust's compile-time ownership lens with runtime, populated-at-scale observation.

## Provenance discipline (unchanged)

Where the paper (or its primary sources) prints a formula, it is used **verbatim and tagged**; where it
is qualitative, it is **operationalized and marked** in-source (the wasp Fermi rules and the wolf score
are verbatim; the termite deposit probability, flocking constants, and the within-tick serialization are
operationalized). Every rung above carries the tags forward — into the `Schedule`'s ordering epochs, into
the metric schema, into the benchmark notes. We invent in the gaps, and we say exactly where.

---
*Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural Multi-Agent Systems,"
Annals of Operations Research 75:69–101 (1997).*
