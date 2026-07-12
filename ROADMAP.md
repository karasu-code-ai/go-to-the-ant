# From "Go to the Ant" (1997) to provable coordination — the Rust edition

This is the **Rust** branch of the polyglot recreation of Parunak's foraging swarm. The other branches ask
their own question (C: *what is the field, physically?*; CUDA: *what if every agent runs at once?*; Go: *what
if each agent is its own process?*). Rust's question is the one that decides whether any of this is
*trustworthy* at scale:

> **Who is allowed to write the field — and can you prove it?**

Everything below is the same swarm, seen through ownership. The shared through-line is preserved; the ladder
and the bridge are rewritten for what Rust's type system and verification tooling can actually deliver.

## The through-line: stigmergy (shared with every branch)

Every system in "Go to the Ant" coordinates the same way — **through a shared, decaying environment**, never
by direct negotiation. Ants don't message each other; they modify a pheromone field and read it back later.
Parunak's word for it (from Grassé) is **stigmergy**: the trace an agent leaves in the world *is* the message,
and evaporation keeps the medium honest — stale information fades, so the collective tracks a moving world
without anyone holding global state.

Three properties fall out of that one idea, and they are why the paradigm survived 25+ years:

1. **The environment is the coordination substrate** — local reads/writes to a field, no lock, no broadcast.
2. **Decay is a feature** — evaporation is a garbage collector for stale coordination.
3. **Emergence over optimization** — the answer is a fixed point of many cheap local updates plus noise.

The intellectual line this sits on: Parunak 1997 → **ACO** (Dorigo, pheromone-on-a-graph), **PSO** (Kennedy &
Eberhart, flocking-as-optimizer), **boids** in graphics/robotics, **ABM platforms** (Swarm, NetLogo, MASON,
Repast), the **actor model** (Hewitt; Erlang/Akka), and now **LLM multi-agent systems** (debate,
self-consistency, tool-using swarms). An ant's pheromone field, an ACO graph, a blackboard, and a shared
scratchpad that many LLM samples read and reinforce are **the same object at different levels of abstraction** —
a decaying, shared medium that turns many cheap local contributions into one global result.

**Provenance discipline** carries through every rung: where the paper (or its primary source) prints a formula
it is used **verbatim** and tagged; where it is qualitative it is **OPERATIONALIZED** and marked. We invent in
the gaps and say exactly where.

## The Rust lens: from ownership to a proof obligation

The other ports *implement* the stigmergic update. Rust turns the update's core safety property into something
the compiler already checks and that external tools can escalate into a proof.

In every current port the pattern is the same: the field is a **single-owner resource**, and the "double
buffered, everyone-sees-the-same-snapshot-this-tick" semantics are **not a convention you have to remember** —
they are a fact the borrow checker enforces.

- **Flocking / wolves** — the step reads the old state through shared `&` borrows and writes the next state
  into a freshly-owned buffer, committed only after the whole phase. No agent can mutate the field another
  agent is still reading; the "simultaneous update" of the reference is made *alias-free and explicit*.
- **Foraging / brood-sort / termites / wasps** — each agent takes a `&mut` borrow of the single-owner
  `World`/`Nest`/`Mound`/`Colony` one at a time in the tick loop, so the stigmergic channel is a strictly
  serial mutation the borrow checker proves has no aliasing.

That is the starting rung: *the implicit serial/simultaneous semantics of the Python reference, made
machine-checked.* The ladder below is about climbing from "the borrow checker is happy" to "we have a proof of
the coordination property that matters."

## The evolution ladder (Rust-specific)

Each rung is a concrete thing a contributor can pick up. They are ordered so each one earns the next.

### Rung 1 — make "who may write the field this tick" a *type-level* invariant

Right now single-owner-per-tick is enforced by *how the loop is written*. Lift it into the types so the
invariant can't be violated even by a future refactor:

- **Type-state the field.** Model the tick as a state machine in types: a `Field<Reading>` exposes only
  `sample(&self, ...)` (shared reads for every agent), and consuming it yields a `Field<Committing>` that
  exposes `deposit(&mut self, ...)`. You cannot deposit while anyone can still read; the phase boundary
  becomes a type transition, not a comment.
- **Or: a typed write-capability token.** Introduce a non-`Clone`, non-`Copy` `WriteCap` that `deposit` demands
  by value/by-`&mut`. "Who may write the field this tick" becomes "who holds the capability" — a single,
  move-only token threaded through the tick. This is deliberately the same primitive that Rung 5 turns into a
  security boundary.

Deliverable: a `stigmergy` module with a `Field` trait (`sample` / `deposit` / `evaporate`) whose *signatures*
encode the read-then-write phase, applied to at least the discrete ports (foraging, termites) without changing
their emergence signatures or their bit-identical outputs.

### Rung 2 — prove there is no UB in the shared-state updates (`miri`)

Once the field ops go through one abstraction, run the whole tick loop under **`cargo +nightly miri`** /
`rustc -Zmiri`. For the current `std`-only, `unsafe`-free ports miri should be *clean by construction* — and
that clean run is worth having as a checked-in artifact, because it is exactly the guarantee that survives the
next rung, when we reach for raw buffers or parallelism. The moment any port adds `unsafe` (a flat `Vec`
backing a 2-D field, an uninitialized next-buffer, a hand-rolled double buffer) miri becomes the tripwire for
aliasing, out-of-bounds, and use-of-uninitialized. Deliverable: a `miri` CI job over every port; a short note
per port stating "no `unsafe`" or "`unsafe` here, miri-checked, why it's sound."

### Rung 3 — property-test the *emergence*, then start model-checking the coordination

Determinism (see the branch's DETERMINISM.md) means these swarms have exact, checkable invariants — perfect
for property-based testing.

- **`proptest` for emergence invariants.** Over random seeds, assert the properties that must hold regardless
  of trajectory: mass is conserved under termite move/deposit; evaporation is monotone non-increasing on a
  cell with no deposit; brood-sort clustering never *decreases* across the run's samples; exactly one wasp
  Chief emerges; total pheromone after a deposit-then-evaporate step obeys the closed-form bound. These are the
  macro-convergence facts from the determinism study turned into `proptest` assertions with shrinking.
- **A step toward `kani`.** For the *discrete* update rule on a small field, use the **Kani** model checker to
  prove bounded properties over *all* inputs, not just sampled seeds: e.g. "on any 4×4 field and any legal move,
  the capability-gated `deposit` conserves total mass and never writes a cell outside the agent's neighborhood."
  Kani turns the Rung-1 capability into something you can *prove* about, not just test. Deliverable: a
  `proptest` suite per port + one Kani harness on the smallest discrete rule as the formal-methods beachhead.

### Rung 4 — fearless *real* parallelism (`rayon`), contrasted honestly with determinism

This is the rung where Rust stops imitating the sequential ports and starts doing what it is uniquely good at —
and where we tell the truth about the cost.

- Parallelize the field-wide, embarrassingly-parallel phases with **`rayon`**: `evaporate` is a
  `par_iter_mut` over cells; the read-only "compute every agent's next state from the old snapshot" phase (the
  flocking/wolves first pass) is a `par_iter().map()` into a fresh buffer. The Rung-1 type-state makes this
  *sound by design*: in the `Reading` phase everything is `&`, so `rayon` parallelizes it with zero data races,
  and the borrow checker guarantees no write escapes into the read phase.
- **The honest contrast.** The deposit phase is where parallelism and determinism collide. Sequential deposit
  gives the bit-identical-to-C trace this branch is known for. A parallel deposit is a concurrent
  read-modify-write — the moment you allow it (atomics, per-shard accumulation, a reduction) you inherit
  CUDA's situation: the *emergence* survives but the exact trajectory does not, because floating-point
  reduction order is not associative. Document this as a first-class modeling choice, not a bug: ship both a
  `deposit_serial` (deterministic, reference-matching) and a `deposit_parallel` (rayon, reproducible only *in
  distribution*), and measure the divergence the same way the flocking study measures cross-language spread.
  This is Rust demonstrating, in one codebase, the determinism study's central claim: *bit-reproducibility is a
  property of the update rule, not the seed.*

### Rung 5 — typed write-capabilities as a security boundary (the polyagentic-security bridge)

Now cash in the capability token from Rung 1. A pheromone field where **deposit demands a capability** is a
model of *who is ALLOWED to write the shared medium* — which is precisely the polyagentic-security question.

- Give the field regions/lanes, and mint capabilities that authorize deposit only into specific ones. A
  **defender** swarm holds capabilities for its lanes; an **attacker** that tries to poison the field, forge a
  trail, or exploit the evaporation rate must *acquire or forge a capability* to do it — and in Rust that
  forgery attempt is a **type error at the boundary**, not a runtime check you hope fires.
- The wolf pack's attraction/repulsion balance (§3.6) is the natural first adversarial testbed: model
  "attacker deposits" vs "defender re-routes" as two swarms sharing one capability-gated field, and use Rungs
  2–3 (miri + proptest/kani) to state and check what is *guaranteed* under attack — e.g. "no agent without a
  lane capability can change that lane's field," proven, not asserted.

## The bridge — verified properties of the shared medium

Two active research directions sit at the end of Parunak's line, and Rust's specific contribution to both is
the same: **making the shared medium something you can prove things about.**

- **Stigmergic distillation** — treat many reasoning samples as a swarm and their reinforced-and-decayed shared
  structure as the pheromone field; a *consensus frontier* emerges that no single sample computed. Rust's job
  here is the trustworthy substrate: a `Field` abstraction (Rung 1) whose read/deposit/evaporate contract is
  type-enforced, `miri`-clean, and `proptest`/`kani`-checked, so the coordination layer under a reasoning swarm
  has *stated, verified* invariants (conservation, monotone decay, single-writer-per-region) rather than hoped-for
  ones.
- **Polyagentic security** — adversarial swarms vs defensive swarms coordinating through a shared environment.
  Rung 5 *is* this bridge: typed write-capabilities make "who is allowed to deposit" a checkable boundary, and
  the formal tooling turns "the attacker cannot forge a trail into a lane it doesn't own" from a claim into a
  proof obligation the compiler and model checker discharge.

Rust is the branch that answers, for the whole polyglot exercise, the question the others can only pose:
**not just "who writes the field," but "prove it."** That is the formal-methods on-ramp to trustworthy
multi-agent coordination.

---
*Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural Multi-Agent Systems,"
Annals of Operations Research 75:69–101 (1997).*
