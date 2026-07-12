# Determinism & divergence — when is a swarm reproducible across languages?

> **Rust's role:** Rust is one of the five *sequential* ports, and it is the strictest twin of C — at a fixed seed it is **bit-identical to C at every seed, on every system**, discrete and chaotic alike, because both link the same platform `libm` and share one `sin`/`cos`/`sqrt`. For the discrete RNG-driven swarms it also bit-matches Go/JS/Java (pinned SplitMix64 + IEEE-754 `+ - * /`); under chaotic flocking it stays glued to C while Go/JS/Java peel off on their own transcendentals. Today Rust buys this determinism with strictly serial, single-`&mut` field updates the borrow checker enforces — rung 4 (rayon) is the point where we deliberately trade that exact-trace reproducibility for real parallelism, and measure the cost honestly.

All six systems run in all six languages share **one PRNG** (SplitMix64, seeded from `--seed`) and the reference's
exact update order and RNG-consumption order. That was deliberate: it turns the 36 ports into an experiment about
*when a multi-agent computation is reproducible* — a question that matters the moment you scale from these toy
swarms to a fleet of computational agents running on different hardware.

The short answer: **reproducibility depends on the system, not just the seed.** Three distinct axes of variation
fall out, and they are easy to conflate.

---

## The setup

At a fixed seed, the five *sequential* ports (C, Rust, Go, JS, Java) execute the *same deterministic computation*:
SplitMix64 emits the same `uint64` stream in every language, they consume it in the same order, and they use only
IEEE-754 double arithmetic (`+ - * /` and comparisons), which is bit-identical across languages. CUDA is the
exception — it runs one thread per agent/cell with per-agent RNG streams and `atomicAdd`/`atomicCAS` writes
(a parallel, Jacobi-style update), so it is distinct by construction.

---

## Axis 1 — across seeds: the real stochastic spread

Run one port across many seeds and the swarm is genuinely noisy. Micro-configuration is highly seed-dependent:

**32-seed sweep (C ports, seeds 0–31):**

| metric | min | max | mean | CV% |
|---|---|---|---|---|
| foraging — deliveries | 33 | 64 | 50.4 | 15.2 |
| brood sort — clustering | 0.842 | 0.979 | 0.900 | 3.4 |
| termites — column count | 3 | 8 | 5.4 | 22.6 |
| termites — tallest column mass | 80,696 | 164,122 | 117,088 | 19.3 |
| wasps — Chief force | 8.45 | 10.22 | 9.36 | 4.4 |
| wasps — # foragers | 2 | 7 | 4.6 | 29.0 |
| flocking — polarization | 0.874 | 0.937 | 0.913 | 1.7 |
| wolves — escape gap° | 63 | 236 | 140.6 | 33.0 |
| wolves — nearest wolf | 0.0 | 6.6 | 0.43 | 270 |

The stochasticity is loud: the tallest termite column varies 2×, the wolf encirclement gap spans 63°–236°, the
wasp forager band is anywhere from 2 to 7. (Wolves' nearest-wolf CV of 270% is a fat tail — the moose is *usually*
pinned to ≈0.1–0.3, but in a few seeds the geometry lets it slip away entirely, hence the 6.6 max. Capture is the
typical outcome, not a guaranteed one.)

**This is the randomness you would expect a MAS to have.** It is fully present.

---

## Axis 2 — across languages, fixed seed, *discrete* systems: bit-identical

Now hold the seed and change the language. For the discrete, RNG-driven systems the five sequential ports produce
the *same number* — at **every** seed, not just one:

**Brood-sort clustering, six languages × eight seeds:**

| seed | C | Rust | Go | JS | Java | CUDA | 5-seq identical? |
|---|---|---|---|---|---|---|---|
| 0 | 0.876 | 0.876 | 0.876 | 0.876 | 0.876 | 0.891 | **yes** |
| 1 | 0.903 | 0.903 | 0.903 | 0.903 | 0.903 | 0.844 | **yes** |
| 2 | 0.913 | 0.913 | 0.913 | 0.913 | 0.913 | 0.932 | **yes** |
| 3 | 0.883 | 0.883 | 0.883 | 0.883 | 0.883 | 0.900 | **yes** |
| 4 | 0.943 | 0.943 | 0.943 | 0.943 | 0.943 | 0.902 | **yes** |
| 5 | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 | 0.914 | **yes** |
| 6 | 0.979 | 0.979 | 0.979 | 0.979 | 0.979 | 0.870 | **yes** |
| 7 | 0.879 | 0.879 | 0.879 | 0.879 | 0.879 | 0.848 | **yes** |

Foraging, brood-sort, termites, and wasps all behave this way. **This cleanliness is *not* convergence or
"the optimization working."** It is the pinned PRNG: the same fixed pseudo-random tape fed to five ports that do
the same arithmetic has nothing to disagree on. Same tape in, same trajectory out.

> Getting here caught a real bug. Java's `randrange` used signed `Math.floorMod`/`%`, which diverges from unsigned
> `next() % n` whenever the PRNG's high bit is set (because `2^64 mod n ≠ 0`). It *looked* fine — the emergence
> still appeared — but silently broke bit-identity for sort and termites. Only the cross-language cross-check
> exposed it (sort/java `0.877` where its siblings said `0.876`). Fixed to `Long.remainderUnsigned`; the numbers
> then snapped into line. **Determinism claims have to be verified empirically, not assumed from per-port tests.**

---

## Axis 3 — across languages, *chaotic* system: divergence returns

Flocking has no `randrange` — every port starts from *identical* initial conditions (its polarization curves agree
exactly for the first two samples). Yet it never bit-matches across languages, at any seed:

**Flocking polarization, six languages × eight seeds:**

| seed | C | Rust | Go | JS | Java | CUDA | 5-seq identical? |
|---|---|---|---|---|---|---|---|
| 0 | 0.898 | 0.898 | 0.913 | 0.844 | 0.934 | 0.925 | no |
| 1 | 0.874 | 0.874 | 0.928 | 0.893 | 0.915 | 0.917 | no |
| 2 | 0.937 | 0.937 | 0.926 | 0.907 | 0.916 | 0.908 | no |
| 3 | 0.904 | 0.904 | 0.897 | 0.913 | 0.931 | 0.909 | no |
| … | … | … | … | … | … | … | no |

Two things to notice. **C and Rust are identical at every seed** — they link the same system `libm`. Go, JS, Java,
and CUDA each peel off — they ship *independent* `sin`/`cos`/`sqrt` implementations, and flocking is chaotic, so
ULP-level differences in those transcendentals amplify exponentially over 600 ticks.

And the magnitude is striking:

- flocking spread **across seeds** (same language): **0.063**
- flocking spread **across languages** (same seed): **0.030–0.090** (0.090 at seed 0)

**Changing the language perturbs a chaotic flock about as much as changing the seed** — at some seeds, more. For a
chaotic system, a floating-point ULP is indistinguishable from a fresh random seed. The *emergence* survives all of
it: every port converges to one coherent flock (polarization ≈ 0.87–0.94). Only the exact trajectory is
non-portable.

---

## Two things that look the same but aren't

The cleanliness a reader senses is really two unrelated phenomena:

1. **Pinned-PRNG determinism** (Axis 2) — cross-language sameness *by construction*, for discrete systems.
2. **Attractor convergence** (the genuine "optimization working") — the *macro* order-parameters self-average to the
   same value regardless of seed: sort → ~0.90 (CV 3.4%), flocking → ~0.91 (CV 1.7%), exactly one Chief always
   emerges, the moose is usually pinned. This is robust emergence, and it is what makes the swarms *useful*.

Meanwhile the *micro*-configuration (Axis 1) stays noisy. **Macro converges; micro stays stochastic.** Neither of
these is the other, and the pinned-PRNG determinism is neither — it is just arithmetic.

---

## CUDA: distinct by design, not divergent by accident

In every parallel system, CUDA's numbers differ from the sequential ports because it reorders the update
(all-agents-read-then-write, `atomicAdd`/`atomicCAS`) and partitions the RNG per agent. For wasps and wolves —
whose face-offs and moose-argmax are irreducibly sequential — the CUDA ports serialize those parts in a
single-thread kernel and mark it `OPERATIONALIZED`. The emergence holds; the trace differs. This is the
concurrent-shared-write reality of running a swarm on real parallel hardware.

---

## The takeaway for scaling to many agents

Whether a multi-agent result is bit-reproducible is a property of **the system**, not the seed:

- **Discrete / event-driven coordination** can be made bit-exact across a heterogeneous fleet — pin the RNG and fix
  the arithmetic/reduction order, and different languages, compilers, and machines agree exactly.
- **Continuous / chaotic coordination** can only be reproduced *in distribution*. A different `libm`, compiler flag,
  or CPU acts like a new random seed; validate on the distribution of outcomes, not on a single trace.
- **Parallelism (GPU)** reorders concurrent writes; reproduce the *distribution*, not the exact trajectory.

For any "field-based" coordination — a pheromone grid, a shared blackboard, a consensus artifact many agents read
and reinforce — the practical rule is: if the field update is discrete and order-fixable, you can get exact
reproducibility across the fleet; if it rides on chaotic continuous dynamics or a nondeterministic parallel
reduction, pin what you can and test on distributions.

---

## Reproduce this

```sh
# across-seed sweep (any language; C shown), any system:
cd c && gcc -O2 -std=c11 -o wolves wolves.c -lm
for s in $(seq 0 31); do ./wolves --seed $s | grep "escape gap"; done

# cross-language at a fixed seed (discrete → identical, flocking → diverges):
for L in c rust go java; do (cd $L && ./sort --seed 3 | grep "final clustering"); done
(cd js && node sort.js --seed 3 | grep "final clustering")
```

*Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural Multi-Agent Systems,"
Annals of Operations Research 75:69–101 (1997).*
