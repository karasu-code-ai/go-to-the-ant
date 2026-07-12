# Go edition — from "Go to the Ant" (1997) to a field of concurrent agents

> This is the **Go** branch's own roadmap (`lang/go`). It keeps the shared through-line — stigmergy as a
> decaying shared medium, and the line from Parunak 1997 to today's two campaigns — but the ladder below is
> Go's: **the path where an agent stops being a struct in a loop and becomes a process.**

## The through-line we keep

Every system here coordinates the same way Parunak's ants do — **through a shared, decaying environment, never
by direct negotiation.** An agent writes a trace into a field and reads it back later; evaporation keeps the
medium honest so stale coordination fades. Three properties carry the whole paradigm:

1. **The environment is the coordination substrate** — local reads and writes to a field, no global state.
2. **Decay is a feature** — evaporation is a garbage collector for stale coordination.
3. **Emergence over optimization** — no agent computes the answer; it is a fixed point of many cheap local updates.

That one object — a pheromone field, an ACO graph, a blackboard, a shared scratchpad many LLM samples read and
reinforce — is the same thing at different levels of abstraction. The intellectual line runs Parunak 1997 →
Ant Colony Optimization → Particle Swarm Optimization → boids in graphics/robotics → the ABM platforms →
**the actor model** (Hewitt 1973; Erlang/Akka) → multi-agent RL → LLM multi-agent systems, and it points at two
campaigns: **stigmergic distillation** (many reasoning samples as a swarm; their reinforced-and-decayed shared
structure as the pheromone field) and **polyagentic security** (adversarial vs. defensive swarms coordinating
through a shared environment).

## Go's lens: *what if each agent is its own process?*

The other ports answer questions about the *field* — what it is physically (C), who may write it (Rust), what
happens when every cell updates at once (CUDA). Go asks the question about the **agent**: not "how is the array
laid out" but "**what is the boundary of one agent, and how do agents that don't share a stack coordinate?**"

Right now the Go ports deliberately *don't* answer it yet. Every system steps one `Agent`/`SortAnt`/`Wolf`
struct sequentially, in index order, over one mutable `World`/`Nest`/`Hunt` store. The actor framing lives in
the **naming** — `Agent`, shared `World`, "communication through the environment" — and the sequential loop is
what keeps Go bit-identical to the other sequential ports. That is the correct rung-0: faithful recreation
first. The Go roadmap is the staircase from *named* actors to *real* ones, and Go is the language built for
exactly this climb — goroutines, channels, `select`, `context`, and a race detector that turns "who owns the
field" from a comment into a checkable property.

The motto for the whole ladder is the Go proverb: **"Don't communicate by sharing memory; share memory by
communicating."** Each rung moves the field one step further behind that principle.

---

## The ladder (Go-specific rungs a contributor can pick up)

### Rung 1 — Goroutine-per-agent, field behind a single owning goroutine

**Build:** turn each agent into its own goroutine and put the pheromone field behind **one owner goroutine**.
No agent touches the grid directly; the owner is the only thing that reads, deposits, and evaporates. This is
the CSP realization of stigmergy: the field is a *server*, agents are *clients*, and the environment is
literally the only shared thing — because it is no longer shared memory at all, it is a process you talk to.

**Concretely, in this repo:** start with `sort.go` or `forage.go`. Replace the `for _, a := range ants { a.step(world) }`
loop with `N` goroutines, each running a `for tick := range clock` loop, all sending sense/deposit requests to
a `field` goroutine over channels. Add a **tick barrier** (`sync.WaitGroup` per tick, or a broadcast on a
`chan struct{}`) so a step is "all agents submit, owner applies, owner evaporates, owner signals next tick" —
this preserves the reference's all-read-then-deposit semantics.

**The honest tension to document:** the moment agents run concurrently, **the update order is no longer fixed by
index**, so bit-identity with the sequential ports breaks *by construction* — the same way CUDA's does. This is
not a bug; it is Go discovering the CUDA lesson from the actor side. Two supported modes, both marked in
provenance comments:
- `--order=sequential` — owner applies agent requests in a canonical order (still bit-matches the sequential ports; the concurrency is only in *sensing*).
- `--order=arrival` — owner applies in channel-arrival order (genuinely concurrent; reproduce the *distribution*, per DETERMINISM.md).

**Verify:** every rung-1 PR must pass `go test -race`. The race detector is the tool that makes "the field is
owned by exactly one goroutine" an enforced invariant rather than a hope.

### Rung 2 — Channels as the sensing/deposit interface

**Build:** freeze the agent↔field contract into a small typed interface so agent and field become independently
swappable — the Go expression of the shared roadmap's "generalize the field" rung.

```go
type Field interface {
    Sense(at Cell) Reading          // local read of the decaying medium
    Deposit(at Cell, amount float64) // reinforce
    // evaporation is the field's own business, ticked by its owner
}
```

Back it two ways behind the same interface: a **direct** in-process implementation (fast, for the single-box
recreations) and a **channel** implementation (`chan senseReq` / `chan depositReq` with reply channels) that is
the actual message-passing substrate. Use `select` with a `context.Context` so an agent that is told to stop
stops cleanly. This is the refactor that later lets a "pheromone" become a *reinforced partial solution* and an
"agent" become a *reasoning sample* without touching the coordination code — the same `Field` interface, a
different payload.

**Concretely:** lift the two-field foraging store (`foodPher`, `homePher`) and the termite `mass`/`scent`
pair behind `Field`. The diffusion/evaporation step becomes a method the owner calls once per tick; agents only
ever `Sense`/`Deposit`. Keep the VERBATIM/OPERATIONALIZED provenance tags on the formulas as they move.

### Rung 3 — A networked field: agents as separate processes

**Build:** cut the last cord. Agents become **separate OS processes / services**, and the field becomes a
**networked shared store** they coordinate through. This is where Go stops modeling a distributed multi-agent
system and *is* one.

Two implementations behind the same `Field` interface from Rung 2:
- **gRPC field service** — `Sense`/`Deposit`/`Tick` as RPCs; the field is a server, agents are clients on other
  machines. Protobuf keeps the wire contract explicit and versioned.
- **Redis-backed field** — the grid as hashes/sorted-sets, deposits as atomic `INCRBYFLOAT`, evaporation as a
  scheduled decay pass or a Lua script. This is the distributed echo of CUDA's `atomicAdd`: the shared field
  under genuine concurrent write contention, now across the network instead of across warps.

**The point:** stigmergy is *already* the right architecture for distribution — because agents never needed to
know about each other, only about the field, you can scatter them across processes and hosts and the
coordination model does not change. A wall appears, the trail re-routes, and no service was reconfigured. That
is the claim the networked field lets you actually demonstrate.

**Reproducibility note (ties to DETERMINISM.md):** a networked, arrival-ordered field is the *chaotic/parallel*
regime — validate on the distribution of outcomes, not a single trace. Keep the discrete systems'
order-fixable path available (canonical apply order on the server) for the cases where bit-exactness across the
fleet is the requirement.

### Rung 4 — Backpressure & fault tolerance: an agent dies, the swarm continues

**Build:** make the field of processes *robust*, which is the property that separates a demo from a system and
the property the actor model was invented for.

- **Backpressure** — bounded channels / bounded RPC concurrency so a fast agent can't drown the field owner;
  `select` with a default or a `context` deadline to shed load. Measure what happens to emergence when deposits
  are dropped under pressure — a stigmergic system should *degrade gracefully*, because evaporation already
  assumes information is lossy.
- **Supervision** — a supervisor goroutine/process that restarts crashed agents (Erlang's "let it crash",
  Go-flavored: `recover` at the goroutine boundary, health-checked worker pools, `errgroup` for coordinated
  shutdown). Kill an agent mid-run and show the trail heals — the field outlives any individual.
- **Partial failure** — the field service itself going down, reconnection, and idempotent deposits so a
  retried `Deposit` doesn't double-count.

**Demonstration:** run the foraging swarm, `kill -9` a third of the agent processes at tick 1500, and show
deliveries keep climbing. That is the headline result of the Go branch.

---

## The bridge to the campaigns (Go's specific on-ramp)

Go's ladder is the **polyagentic-systems** path. The other branches reach the campaigns through the field's
physics; Go reaches them through the *agents' autonomy*.

**Toward stigmergic distillation.** Rungs 1–3 turn "many reasoning samples sharing a scratchpad" from a
metaphor into an architecture: each reasoning agent is a process, the consensus frontier is a networked field
(the Rung-2 `Field` interface with a "reinforced partial solution" payload), good partial reasoning is a
`Deposit`, and evaporation is the decay that stops the swarm from ossifying on an early wrong path. Go's
contribution is that these agents can be **genuinely independent processes** — different models, different
tools, different hosts — coordinating only through the shared frontier, which is exactly what a real
multi-sample reasoning system looks like at scale.

**Toward polyagentic security.** This is where Go's rung-4 substrate is the *point*. A polyagentic security
setting is adversarial agents that poison the field, exploit the evaporation rate, or forge trails, versus
defensive agents that detect and re-route — the wolf-pack attraction/repulsion balance (Parunak §3.6), one
level up. The actor model's **supervision and isolation** are the security primitives: process boundaries mean
one compromised agent can't reach into another's memory; the single-owner field means every write goes through
one auditable choke point where you can rate-limit, authenticate, and log deposits; backpressure is your
defense against a flooding adversary. Rust's branch proves properties about *who may write the field*; Go's
branch builds the *runtime isolation and supervision* that contains an agent that writes it maliciously. Same
question — "who owns the shared field and who is allowed to write it" — answered with process isolation instead
of the type system.

---

## Provenance discipline (unchanged across the branch)

Where the paper or its primary sources print a formula, it is used **verbatim** and tagged (the two wasp Fermi
formulas; Korf's `S = d(moose) - k·d(nearest wolf)`; Reynolds' three rules; Deneubourg's pick/drop
probabilities). Where it is qualitative, it is **operationalized** and marked (termite deposit probability;
wolf continuous-plane candidate search and `k=1.12`; boid radii and weights). Every new rung inherits this: when
a deposit becomes an RPC or a Redis `INCRBYFLOAT`, the formula it carries keeps its tag, and any concurrency- or
network-forced deviation from the reference update order is reported as such — the same way the sequential
face-off in `wolves.go` and the arrival-ordered field in Rung 1 are.

---
*Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural Multi-Agent Systems,"
Annals of Operations Research 75:69–101 (1997).*
