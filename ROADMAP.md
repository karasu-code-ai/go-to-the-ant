# From "Go to the Ant" (1997) to modern multi-agent systems — the JavaScript edition

This is the JavaScript branch of a polyglot recreation of Parunak's foraging swarm. The other branches ask
what the field *is* (C), what happens when every agent runs at once (CUDA), who is *allowed* to write it (Rust),
and what if each agent is its own process (Go). **JavaScript asks: who gets to *see* it?** — and answers by
being the accessible front door, the one lens that turns the whole swarm into something you can watch, tweak,
and share behind a single link, with nothing to install.

This document keeps the shared through-line and then charts JavaScript's own path forward. It is meant to be
picked up: every rung below is a concrete build a contributor could start this week.

## The through-line: stigmergy (shared across every language)

Every system in "Go to the Ant" coordinates the same way — **through a shared, decaying environment**, never by
direct negotiation. Ants don't message each other; they modify a pheromone field and read it back later.
Parunak's word for it (borrowed from Grassé) is **stigmergy**: the trace an agent leaves in the world *is* the
message, and evaporation is what keeps the medium honest — stale information fades, so the collective tracks a
moving world without anyone holding global state.

Three properties fall out of that one idea, and they are the whole reason the paradigm survived 25+ years:

1. **The environment is the coordination substrate.** No shared memory to lock, no broadcast to synchronize —
   just local reads and writes to a field.
2. **Decay is a feature.** Evaporation is a built-in garbage collector for stale coordination. It is why the
   trail re-routes around a new wall and why depleted sources are forgotten.
3. **Emergence over optimization.** No agent computes the answer. The answer is a fixed point of many cheap
   local updates plus noise.

The straight line to draw: an ant's pheromone field, an ACO graph, a blackboard architecture, and a shared
scratchpad that many LLM samples read and write are **the same object at different levels of abstraction** — a
decaying, shared medium that turns many cheap local contributions into one global result. Parunak 1997 →
Ant Colony Optimization (Dorigo) → Particle Swarm Optimization (Kennedy & Eberhart) → boids in graphics and
robotics (Reynolds) → agent-based-modeling platforms (Swarm/NetLogo/MASON/Repast) → the actor model (Hewitt;
Erlang/Akka) → multi-agent RL → today's **LLM multi-agent systems** (debate, self-consistency, tool-using
swarms). Same object, rising abstraction.

## What JavaScript is, as a lens

The JS ports are **zero-dependency, standard-library-only**, run under Node, and are built to be lifted straight
into a browser. That constraint is the whole point: the field is portable to any device with a browser, so the
swarm becomes *legible to anyone*. Two design decisions make it a serious port and not a toy:

- **`BigInt` carries the 64-bit RNG.** JavaScript has no native `uint64`, so SplitMix64 is implemented in exact
  `BigInt` integer math and consumed in the reference's exact order. That is why the discrete systems
  (foraging, brood-sort, termites, wasps) are **bit-identical** to the C, Rust, and Go ports at every seed — the
  same pseudo-random tape fed to the same arithmetic has nothing to disagree on. The simulation math itself runs
  on ordinary IEEE-754 doubles.
- **Provenance is marked in-source, same discipline as every branch.** Where the paper prints a formula it is
  used VERBATIM and tagged (the wasp Fermi win probability `1/(1+e^(h(Fi-Fj)))`, the wolf score
  `S = d(moose) - k*d(nearest wolf)`); where it is qualitative it is OPERATIONALIZED and tagged (boid weights and
  radii, the wolves' continuous plane + 24-candidate search). We invent in the gaps and say exactly where.

The current surface: `forage.js`, `sort.js`, `termites.js`, `wasps.js`, `wolves.js`, `flocking.js`, each a Node
CLI with an ASCII renderer and a scalar emergence signature. That ASCII renderer is the seed of everything below —
it is already a *view* of the field; the roadmap is about making that view real, authentic, and shareable.

## The JavaScript ladder (the divergence — where this branch goes next)

The other branches climb toward the campaign by getting closer to the metal (CUDA, Rust ownership, Go processes).
JavaScript climbs by getting closer to the *viewer*. Each rung makes the swarm more directly inspectable while
keeping the shared field abstraction intact.

### Rung 1 — One engine behind both the CLI and the visualizer

Today each script owns its own loop, RNG, field, and renderer. The refactor: extract a single
**`stigmergy-core`** ES module that every system imports — a `Field` (backing store + `deposit` / `read` /
`evaporate` / `diffuse`), an `Agent` step contract, a `SplitMix64` (`BigInt`), and a headless `run(config)` that
emits per-tick state. The Node CLI and the browser visualizer then become two thin front-ends over the *same*
engine, so a number you see animate in the browser is provably the number the CLI prints.

Concrete JS moves:
- Back the field with **typed arrays** (`Float64Array` / `Int32Array`) instead of nested arrays — a flat,
  cache-friendly buffer that is also exactly what you hand to a GPU or a WASM module later. This is the interface
  the shared spine calls "generalize the field": one `Field` abstraction, agent and medium swappable.
- Make the renderer a pluggable sink: an ASCII sink for the terminal, a **Canvas 2D** sink for the browser (one
  `putImageData` from the same typed-array buffer, no per-cell DOM). Same state in, different pixels out.
- Ship it as a plain ES module with **no build step** — `import` in Node 20+ and in the browser unchanged. The
  zero-dependency rule is a feature: the accessible front door can't demand a toolchain.

*Ships:* a live Canvas visualizer that runs the authentic JS swarm at 60fps, sharing one code path with the
verified CLI. Accessibility work rides here too — a colorblind-safe pheromone ramp, a text-mode fallback that
reuses the ASCII sink, and `prefers-reduced-motion` support so the emergence is legible to everyone.

### Rung 2 — WASM: run the *authentic compiled swarm* in the browser

A JS reimplementation of the swarm is convincing but it is still a reimplementation. The stronger claim: put the
**actual C / Rust / Go port** on screen, byte-for-byte the binary the determinism study measured, compiled to
**WebAssembly** and driven by the JS visualizer.

- Compile the **C** port with Emscripten, the **Rust** port with `wasm-bindgen` / `wasm-pack`, the **Go** port
  with **TinyGo** (`GOOS=wasm`) — each exposing the same tiny ABI: `init(seed)`, `step()`, and a pointer to the
  field buffer. JS reads that buffer directly out of WASM linear memory as a typed-array view — **zero-copy** —
  and blits it to Canvas. The swarm computes in native-compiled code; JS only paints.
- This makes the visualizer a **cross-language witness**: run the C-WASM foraging swarm and the JS foraging swarm
  side by side at the same seed and watch them stay bit-identical, then switch to flocking and watch them *visibly
  diverge* — because C's `libm` and V8's `Math` disagree at the ULP and the flock is chaotic. The DETERMINISM
  finding becomes something you *see*, not just a table you read.
- **Web Workers** keep each WASM swarm off the main thread; `SharedArrayBuffer` (with the COOP/COEP headers) lets
  the render thread read the field while the worker steps it — the same "many local writers, one shared medium"
  story, now literally across threads.

*Ships:* the browser as an authenticity layer. Anyone can load a page and confirm the reproducibility claim for
themselves against the real compiled artifacts — reproducible-in-a-link.

### Rung 3 — WebGPU: a browser-native GPU swarm (the CUDA lens, portable)

CUDA is the parallel lens but it needs a specific GPU and a toolchain. **WebGPU** brings the same substrate to any
modern browser. This is JavaScript's version of the "parallelize" rung: the pheromone field as a GPU storage
buffer, deposits as concurrent atomic writes, the update as an all-agents-read-then-write Jacobi step.

- Port the field to a `GPUBuffer`; write the step as a **WGSL** compute shader — one invocation per agent or per
  cell, `atomicAdd` on the pheromone buffer for deposits under contention, a second pass for evaporation/diffusion.
  This mirrors the CUDA port's design exactly, which is the point: same parallel semantics, portable target.
- Be honest about the same divergence CUDA has: reordering concurrent writes and using per-agent RNG streams makes
  the trajectory differ from the sequential ports by construction. Mark it OPERATIONALIZED, reproduce the
  *distribution* not the trace — the DETERMINISM finding's rule for parallel reductions applies unchanged.
- Render from the same buffer with zero round-trips: the compute pass and the fragment/blit pass share the
  `GPUBuffer`, so the swarm never leaves the GPU between step and paint. This is where the JS branch can push
  agent counts from hundreds to hundreds of thousands and make the *scale* of emergence visible.

*Ships:* a portable GPU swarm anyone can run without CUDA hardware — the compute pattern of "a swarm of samples on
a GPU" demonstrated in a tab.

### Rung 4 — The interactive, shareable playground

The payoff rung, and the one only JavaScript can build. A page where the rules are live controls and the entire
run state is encoded in the URL.

- **Live rule editing:** sliders for evaporation rate, deposit strength, diffusion, agent count, perception radius,
  the boid weights, the wolf `k`. Change one and the field responds in the next frame — you *feel* why decay is a
  feature by turning it to zero and watching the trail stop forgetting.
- **Shareable seeds:** the seed and full config serialize into the URL (query string or hash), so a link *is* a
  reproducible experiment. "Watch this exact flock at this exact seed diverge from the C port" becomes a link you
  paste into an issue. Add a deterministic replay (record the config, not the frames) and one-click **GIF/WebM
  capture** for embedding results.
- **Inspection surface:** hover a cell to read its pheromone value; scrub a timeline; overlay the scalar emergence
  signature (deliveries, clustering, polarization, column count) as a live sparkline next to the field. This turns
  the visualizer into a debugger for a multi-agent system's behavior.

*Ships:* the swarm and distillation ideas made legible and reproducible-in-a-link for anyone — a working
instrument for inspecting how a decaying shared medium produces collective structure.

## The bridge to the campaign (same destination, JavaScript's on-ramp)

Two active research directions sit at the end of the stigmergy line, and the JS branch is the **accessible,
inspectable surface** for both — framed here at the same conceptual level as the shared roadmap.

- **Stigmergic distillation.** Treat many independent reasoning samples as a swarm and their shared,
  reinforced-and-decayed intermediate structure as the pheromone field: good partial reasoning gets reinforced and
  followed, weak paths evaporate, and a *consensus frontier* emerges that no single sample computed — the foraging
  trail, one level up. Because Rung 1 makes the `Field` abstraction agent-agnostic, an "agent" can become a
  reasoning sample and a "pheromone" a reinforced partial solution *in the same engine*. JavaScript's contribution
  is the **view**: a browser surface that renders the consensus frontier forming and lets a researcher watch which
  partial paths reinforce and which evaporate — the emergence made watchable in a link.

- **Polyagentic security.** A swarm is a threat model *and* a defense: adversarial agents that coordinate through a
  shared environment (poisoning the field, exploiting the evaporation rate, forging trails) versus defensive swarms
  that detect and re-route — the same attraction/repulsion balance as the wolf pack. The JS playground is a natural
  **red-team/blue-team console**: pit an attacker swarm against a defender swarm in the shared field and *see* the
  poisoning and the re-routing happen, with every parameter live and every scenario shareable as a URL. The
  ownership/formal guarantees live on the Rust branch; JavaScript is where you *demonstrate* what those guarantees
  are protecting.

## Provenance discipline (carried through every rung)

Where the paper (or its primary sources) prints a formula, it is used verbatim and tagged; where it is qualitative,
it is operationalized and marked in-source. That discipline does not relax when the field moves to Canvas, WASM, or
WebGPU — a WGSL shader that reorders writes is marked OPERATIONALIZED exactly as the CUDA kernel is, and the
visualizer surfaces those tags so a viewer can see which numbers are the paper's and which are ours.

---
*Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural Multi-Agent Systems,"
Annals of Operations Research 75:69–101 (1997).*
