# "Go to the Ant" — six emergent swarms, recreated from the paper

> **Branch `lang/go` — the Go edition** (agents as goroutines over a shared store). This branch adds a faithful, dependency-free Go port
> of all six swarms in [`go/`](./go/), beside the pure-Python originals it mirrors. Each language lives on its own
> `lang/*` branch; the shared story is in [ROADMAP.md](ROADMAP.md) and the cross-language reproducibility findings
> in [DETERMINISM.md](DETERMINISM.md).

Faithful, from-scratch recreations of all six natural multi-agent systems in Parunak's keystone stigmergy
paper — ant **foraging**, ant **brood-sorting**, termite **nest-building**, wasp **caste-differentiation**,
bird/fish **flocking**, and wolf-pack **pursuit**. Each is a handful of local rules from which global order
*emerges* — no leader, no plan, no central control. Pure Python (standard library only) + a live interactive
web visualizer.

**Source:** H. Van Dyke Parunak, *"'Go to the Ant': Engineering Principles from Natural Multi-Agent Systems,"*
Annals of Operations Research 75:69–101 (1997) — the keystone paper of stigmergic multi-agent design.

## Quick start
No dependencies, no build, no install — just Python 3.8+.

```bash
python3 go_to_the_ant.py        # ant foraging: a nest↔food trail self-assembles (try --wall)
python3 brood_sorting.py        # scattered items self-sort into clusters
python3 termites.py             # deposits self-concentrate into columns
python3 wasps.py                # identical wasps split into Chief / Foragers / Nurses
python3 flocking.py             # random headings cohere into one flock
python3 wolves.py               # six wolves surround the moose, no comms

bash demo.sh                    # ...or run a short guided tour of all six
```

**The live version** — open **`visualizer.html`** in any browser (self-contained, works offline; if your
browser blocks local-file scripts, run `python3 -m http.server` and visit `localhost:8000/visualizer.html`).
Six tabs, live controls, watch each swarm self-organize in real time.

## What it is
`go_to_the_ant.py` implements the paper's §3.1 foraging colony as **five local rules per ant** + one field law,
nothing more:

1. **Avoid obstacles.** An ant never steps into a wall.
2. **Wander, biased by pheromone.** No scent → Brownian motion (uniform over the 8 directions). Scent present
   → same random walk, but the direction distribution is *weighted* toward the pheromone.
3. **Carry ⇒ deposit.** A food-carrying ant drops pheromone at a constant rate as it walks (and heads for a
   home beacon — the paper notes beacon and pure-wander "yield the same global behavior," beacon just sooner).
4. **At food, empty-handed ⇒ pick it up.**
5. **At the nest, carrying ⇒ drop it** (a delivery).

Plus the field law: the trail **diffuses a little AND evaporates every tick.** Spreading gives the pheromone
"some breadth" so nearby sub-trails "merge together into a trace" (Parunak §3.1); evaporation fades the rest.
No ant plans a route. The nest↔food path — a minimum spanning tree in the limit (Goss et al. 1989) — *emerges*
from deposit + diffusion + evaporation + weighted-random following.

## What emerges (results)
```
python go_to_the_ant.py --ticks 4000 --ants 110 --seed 1
  deliveries(t): 0 0 0 0 0 1 9 15 22 29 38 42 47 57 65 71 77 80 87 93 → 101 total
python go_to_the_ant.py --ticks 5000 --ants 130 --seed 3 --wall
  deliveries(t): 0 0 0 0 1 3 5 7 9 14 26 32 40 51 65 74 88 100 112 126 → 137 total (routes the gap)
```
The delivery curve is the **S-curve signature of stigmergic emergence**: a flat start (Brownian search for the
first food), then acceleration as the first returning carriers lay a trail and *positive feedback* recruits more
searchers onto it ("**cultivate increasing returns**" — the paper's own principle), then steady foraging. The
ASCII render shows the trail bridging N↔F; with `--wall`, it bends to thread the single gap — obstacle routing
with zero global planning.

## The one real bug — and why it's the whole lesson
First version: **0 deliveries.** Ants reached food, picked it up, and never came home — all of them trapped
near the food. The cause: carrying ants were following *their own fresh pheromone deposits* (scent weight ≫ the
weak homing pull), so they circled in the cloud they were laying. The fix is faithful to the paper and *is* the
mechanism: **searchers READ the trail (rule 2, scent-biased); carriers WRITE it (rule 3, head for the beacon,
don't chase scent).** Separating read from write is what makes stigmergy work — and it is a failure mode that
recurs throughout multi-agent systems: following your own signal is not the same as independent information.

## Design principles from the paper, honored here
- **Agents small** — five rules, local sensing (the 8 neighbors), short reach.
- **Decentralized / share through the environment** — the pheromone field is the *only* channel; no ant talks
  to another.
- **Support entropy** — the Brownian floor never switches off, even on a strong trail; that residual wandering
  is what cuts short-cuts across the initial meanders (the paper's straightening effect).
- **Pursue no optima** — no ant computes or knows the shortest path; the MST is an emergent side-effect.
- **Persistent disequilibrium** — evaporation keeps the field decaying, so the trail must be continually
  re-earned by deliveries or it fades.

## Run it
```
python go_to_the_ant.py                        # default ASCII demo
python go_to_the_ant.py --wall                 # add a wall with a gap (routing demo)
python go_to_the_ant.py --ticks N --ants M --seed S --png trail.png   # (--png needs matplotlib)
```
Pure Python + stdlib — no dependencies. Recreated from the paper alone, as a study of the swarm that started it all.

## Companion: the double bridge — `double_bridge.py` + `double_bridge.html`
Goss et al. 1989's actual experiment. Two branches of *different length* join nest and food, and the colony
**converges on the shorter one over time** — the short branch returns sooner, so it re-deposits sooner, and the
nonlinear choice `p ∝ (c+φ)²` amplifies that lead. This is the path *optimization* the open-plane forager can't
show, because there is no shorter route to *choose*. Run `python3 double_bridge.py` for the convergence curve,
or open `double_bridge.html` to watch the colony shift onto the short branch.

## §3.2 Brood sorting — `brood_sorting.py`
The second colony (Deneubourg et al. 1991): scatter three kinds of brood items at random; four local rules
(wander, short memory, `p(pickup)=(k₊/(k₊+f))²`, `p(putdown)=(f/(k₋+f))²`, with k₊=0.1 < k₋=0.3 —
Deneubourg's actual constants — and a ~15-step memory) make like items cluster with no ant ever comparing the
whole nest. Emergent sorting: clustering climbs **0.35 → ~0.95**;
three clean type-clusters appear. As the paper warns, k₋ must exceed k₊ or clusters dissolve faster than they
form, and too-long a memory makes the ant "see the whole nest as one place" and sorting stops.

## Live visualizer — `visualizer.html`
An interactive canvas recreation of **all six systems** (six tabs) — watch each swarm self-assemble in real
time, with per-system controls (population / evaporation / speed, and a wall-with-a-gap toggle for foraging).
Pheromone renders as an amber heat field; searchers are cream, carriers glow.

## The full "Go to the Ant" zoo (all six systems)
Reading the whole paper surfaced **six** natural multi-agent systems — this repo recreates all of them, each a
faithful, from-scratch port of one section:
- **§3.1 Ants — foraging** ✅ (`go_to_the_ant.py`) — pheromone path-planning → minimum spanning tree.
- **§3.2 Ants — brood sorting** ✅ (`brood_sorting.py`) — memory-modulated pickup/putdown → clustering.
- **§3.3 Termites — nest building** ✅ (`termites.py` + visualizer tab) — wander toward strongest pheromone;
  deposit ∝ local density + load. The scent field both **diffuses** (a local Brownian stencil) **and decays**
  (§4.6), so fresh pile-cores win while each pile gains some **breadth** → scattered dabs self-concentrate into
  distinct COLUMNS (nucleate→consolidate, e.g. 55→27→17 in the visualizer). Arches/floors are the next rung.
- **§3.4 Wasps — task differentiation** ✅ (`wasps.py` + visualizer tab) — Force + Foraging-Threshold per wasp;
  stochastic Fermi-function face-offs transfer force; brood Demand dynamics. Genetically identical wasps self-sort
  into Chief / Foragers / Nurses with no HR department (a NON-spatial emergence).
- **§3.5 Birds & fish — flocking** ✅ (`flocking.py` + visualizer tab) — Reynolds' three rules → one coherent
  flock (polarization 0.03→0.93). Key fix: NORMALIZE the three urges or cohesion's position-scale swamps alignment.
- **§3.6 Wolves — surrounding prey** ✅ (`wolves.py` + visualizer tab) — moose flees; wolves minimise
  S = d(moose) − k·d(wolf); k≈1.12 → the pack closes on the moose in the open. ALL SIX SYSTEMS DONE.

The §2 theory (agents / environment / homodynamic vs heterodynamic coupling) and §4 principles (Holland's
Aggregation/Nonlinearity/Flows/Diversity + Kelly's maxims) are the scaffolding these six hang on.
