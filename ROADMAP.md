# From "Go to the Ant" (1997) to modern multi-agent systems — a polyglot evolution

This directory recreates Parunak's foraging swarm in several programming languages, **not** as a syntax
parade but as a set of *lenses*. Each language forces you to answer a different question about how the system
actually works, and those questions are the same ones you have to answer when you scale from ants to
thousands of coordinating computational agents.

## The through-line: stigmergy

Every system in "Go to the Ant" coordinates the same way — **through a shared, decaying environment**, never
by direct negotiation. Ants don't message each other; they modify a pheromone field and read it back later.
Parunak's word for it (borrowed from Grassé) is **stigmergy**: the trace an agent leaves in the world *is* the
message, and evaporation is what keeps the medium honest — stale information fades, so the collective tracks a
moving world without anyone holding global state.

Three properties fall out of that one idea, and they are the whole reason the paradigm survived 25+ years:

1. **The environment is the coordination substrate.** No shared memory to lock, no broadcast to synchronize —
   just local reads and writes to a field.
2. **Decay is a feature.** Evaporation is a built-in garbage collector for stale coordination. It is why the
   trail re-routes around a new wall and why depleted sources are forgotten.
3. **Emergence over optimization.** No agent computes the answer. The answer (a minimum-spanning-tree trail,
   a sorted brood, a surrounded moose) is a fixed point of many cheap local updates plus noise.

## What happened since 1997 (the shoulders we build on)

Parunak was writing a *survey of principles*, pointing at biology to argue for a style of engineering. The
quarter-century since turned each of his six vignettes into its own field:

- **Ant Colony Optimization** (Dorigo, 1992→) turned the foraging trail into a general combinatorial optimizer —
  pheromone-on-a-graph solving TSP, routing, scheduling. This is the direct descendant of §3.1.
- **Particle Swarm Optimization** (Kennedy & Eberhart, 1995) took the flocking rules (§3.5) and made them a
  continuous-space optimizer.
- **Boids in graphics & robotics** — Reynolds' rules (§3.5) became the backbone of crowd simulation and drone
  swarm control.
- **Agent-Based Modeling platforms** — Swarm, NetLogo, **MASON**, Repast (mostly JVM, which is *why* Java is one
  of our lenses) made "many local agents in a shared field" a first-class scientific instrument.
- **The actor model** (Hewitt 1973; Erlang/Akka in practice) generalized "an agent is a process that only reacts
  to its local mailbox" — the message-passing cousin of stigmergy, and the reason **Go/goroutines** is a lens.
- **Multi-agent reinforcement learning** and, most recently, **LLM multi-agent systems** (debate, self-consistency,
  tool-using agent swarms) put *reasoning* agents into the same coordinate-through-a-shared-artifact pattern.

The straight line to draw: an ant's pheromone field, an ACO graph, a blackboard architecture, and a shared
scratchpad that many LLM samples read and write are **the same object at different levels of abstraction** — a
decaying, shared medium that turns many cheap local contributions into one global result.

## Where this overlaps with the current work

Two active research directions sit at the end of that line, and this polyglot exercise is the on-ramp to both:

- **Stigmergic distillation** (the reasoning-model line of work). Treat many independent reasoning samples as a
  swarm and treat their shared, reinforced-and-decayed intermediate structure as the pheromone field: good
  partial reasoning gets reinforced and followed, weak paths evaporate, and a *consensus frontier* emerges that
  no single sample computed — exactly the foraging trail, one level up. The CUDA port is not a toy here: the
  pheromone field as **shared global memory** and deposits as **atomic read-modify-write under contention** is
  the literal compute pattern of running a swarm of samples on a GPU.

- **Polyagentic security.** A swarm is a threat model *and* a defense. Adversarial agents that coordinate through
  a shared environment (poisoning the field, exploiting the evaporation rate, forging trails) versus defensive
  swarms that detect and re-route — the same attraction/repulsion balance as the wolf pack (§3.6), the same
  "who owns the shared field and who is allowed to write it" question that **Rust's** ownership model makes
  explicit and that formal tools (miri, model checking) let you actually *prove* things about.

## The language lenses

| Language | The question it forces | What it teaches for scaling to many agents |
|---|---|---|
| **C** | What is the field, physically? | A raw array. Stigmergy with nothing hidden — the baseline mental model. |
| **CUDA** | What if every agent runs at once? | Field = global memory; deposit = `atomicAdd` under contention; the update order (all-read-then-write) is a real, principled change. The GPU-swarm substrate. |
| **Rust** | Who is allowed to write the field? | Ownership/borrowing make the shared-mutable-state question un-ignorable. The formal-verification and security lens. |
| **Go** | What if each agent is its own process? | Ants → goroutines, field → shared store/channel. The step toward genuine actor-style multi-agent systems. |
| **JavaScript** | Who gets to see it? | Zero-dependency, browser-portable — the accessible, visual lens that plugs into the live visualizer. |
| **Java** | How did the field's scientists build it? | The MASON/Repast/NetLogo ABM lineage — the classical platform view. |

## The evolution ladder (how each round builds on the last)

1. **Recreate faithfully** (done in Python; this round: 6 languages, all verified). Same emergence, one system.
2. **Parallelize** — the CUDA port makes the sequential→concurrent update order an explicit modeling choice, and
   surfaces the concurrent-write (`atomicAdd`) reality of a real swarm. Extend to the other five systems.
3. **Generalize the field** — abstract "a decaying shared medium with local read/deposit/evaporate" into one
   interface, so the *agent* and the *field* become swappable. This is the refactor that lets an "agent" become
   a reasoning sample and a "pheromone" become a reinforced partial solution.
4. **Swap in reasoning agents** — replace the random-walk ant with a sample/policy; keep the stigmergic
   scaffold. This is the bridge to the distillation line.
5. **Adversarial swarms** — add agents that attack the field and agents that defend it; use the ownership/formal
   lenses to reason about what can be guaranteed. The bridge to the security line.

Provenance discipline carries through every rung: where the paper (or its primary sources) prints a formula, it
is used verbatim and tagged; where it is qualitative, it is operationalized and marked. We invent in the gaps —
and say exactly where.

---
*Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural Multi-Agent Systems,"
Annals of Operations Research 75:69–101 (1997).*
