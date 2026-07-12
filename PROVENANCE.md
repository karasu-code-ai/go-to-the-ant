# Provenance & verification — the "Go to the Ant" zoo

What in each recreation is **verbatim from Parunak's paper**, what is **operationalized** (the paper is
qualitative / gives no numbers), and what free parameters were **tuned** to reproduce the claimed emergence.
Plus the **primary sources to verify against** for a full fidelity pass.

Guiding honesty: where the paper prints a formula, we use it exactly. Where it lists a rule but no number,
we invented a form faithful to the stated dependencies and *say so at the line* (`# OPERATIONALIZED …`).

---

## Per-system audit

| § | System | VERBATIM from paper | OPERATIONALIZED (paper qualitative) | TUNED free params | Emergence reproduced |
|---|---|---|---|---|---|
| 3.1 | Foraging | the 5 rules; evaporation; "weighted toward scent" | scent weight ×7, homing pull ×4, region sizes | ants, evaporation | ✓ trail = MST; S-curve deliveries; routes a wall gap |
| 3.2 | Brood sorting | **`p(pickup)=(k₊/(k₊+f))²`, `p(putdown)=(f/(k₋+f))²`, k₊=1<k₋=3, mem≈10** — all printed | (nothing beyond the printed formulas) | ants, ticks | ✓ clustering 0.35→0.92 |
| 3.3 | Termites | the 3 rules + the 3 stated dependencies | **the whole deposit `p` — paper gives NO formula** | decay, metab, appetite | ✓ scattered dabs → distinct columns |
| 3.4 | Wasps | **face-off `p=1/(1+e^(h(Fᵢ−Fⱼ)))`, forage `p=1/(1+e^(h(σⱼ−D)))`, σ learn/forget, D recurrence** | work `W`=count of mobile foragers (paper: "work by foragers", no formula); force-cap; force→mobility gate | appetite, k-caps, h | ✓ 1 Chief + ~7 Foragers + ~72 Nurses |
| 3.5 | Flocking | the 3 Reynolds rules (separation/alignment/cohesion) | perception radius, sep distance, 3 weights, "normalize each urge" | weights, radius | ✓ polarization 0.03→0.93 |
| 3.6 | Wolves | the 2 rules + **score `S = d(moose) − k·d(wolf)`** | speeds, k, continuous plane (paper: hex grid) | vm, vw, k=1.12 | ✓ pack surrounds/pins the moose |

**Biggest inventions (verify hardest):** the termite deposit probability (§3.3 — no formula in the paper at
all) and the wasp work/force-bounding (§3.4 — I needed a force cap + mobility gate + appetite tuning the
paper doesn't specify; the two Fermi functions are the paper's).

---

## Primary sources to verify against

Parunak cites these for the actual models; the "Go to the Ant" paper only summarizes them. To check our
operationalizations against the real math, we'd need:

| System | Primary source (as cited) | Why we need it |
|---|---|---|
| 3.1 foraging | **Goss, Aron, Deneubourg & Pasteels 1989** ("Self-organized shortcuts…"); Steels 1991 | the MST result + the exact 5-rule form |
| 3.2 sorting | **Deneubourg, Goss, Franks, Sendova-Franks, Detrain & Chrétien 1991** ("The dynamics of collective sorting") | confirm k₊/k₋ and the memory model (we already match the printed formulas) |
| 3.3 termites | **Kugler & Turvey 1990** (or the Deneubourg termite model) | the actual deposit-probability formula — our biggest invention |
| 3.4 wasps | **Theraulaz, Goss, Gervet & Deneubourg 1991** (response-threshold / polyethism) | the real `W`, demand-growth `d(t)`, and force dynamics — replaces our tuning |
| 3.5 flocking | **Reynolds 1987** ("Flocks, herds and schools", SIGGRAPH); Heppner 1990 | Reynolds' tuned weights/radii |
| 3.6 wolves | **Korf 1992** ("A simple solution to pursuit games"); Manela & Campbell 1995 | the hex-grid "six wolves always capture" proof + k |

None of these primary sources were consulted during this recreation — it was built from Parunak's summary paper alone. Acquiring them is the natural next fidelity pass.

---

## Applying the Engineering Principles to our own recreations (2026-07-12)
A first "feet-wet" pass.

### ✅ Foraging — removed a genuine principle violation (§4.3.3 Small in Scope + §4.6 multi-marker)
The v1 carrier used a **global homing beacon** (it computed the direction to the nest — non-local knowledge
the paper flags as merely optional). Replaced with **two local pheromone fields**: carriers lay the FOOD
trail and follow a HOME field that the nest **broadcasts and diffuses** (§4.6 dissipation → gradient). Now
*no ant knows where the nest is* — it climbs the local home gradient. **Result: same emergence, arguably
better** — 98 deliveries (was 95); wall variant 142 (was 117 — the diffused gradient routes the gap
naturally). A clean win: more principled AND more faithful AND better.

### ⚠️ Wasps — applied the entropy leak, learned a coupling lesson (§4.6)
The v1 force dynamics needed an **ad-hoc force CAP** to stop one super-wasp. §4.6 explicitly names the
wasp force-flow an entropy leak, so we replaced the cap with **force dissipation + regeneration** (`leak`,
`gen`) — a principled bound (equilibrium ≈ gen/leak; Fmax settles ~13 not ∞). **It keeps the caste COUNTS
(1 Chief + ~8 Foragers + ~71 Nurses) and the FORCE ordering.** BUT the Chief's *high-threshold* detail
regressed: dissipating the force reshapes the force distribution, which is **coupled** to the demand/threshold
balance, and we could not recover the idle-high-σ Chief by tuning appetite/equilibrium alone. **The lesson
(a real one): the force-flow and threshold-response mechanisms are coupled and can't be
tuned independently** — exactly the kind of cross-mechanism interaction the principles warn lives in these
systems. Honest partial win: principled force-bounding, at the cost of a secondary fidelity feature.

**RESOLVED (2026-07-12).** The Chief's high-threshold needs the paper's *spatiality* — the Chief "wanders and
faces off," so it's not near the brood and rarely stimulated, so its σ rises. Added a spatiality PROXY:
`dominance=(F/Fmax)^4` suppresses the top wasp's foraging (only F≈Fmax, i.e. the Chief), leaving the foragers
untouched. All three castes now emerge faithfully across seeds — **Chief high-F + high-σ (2.7–4.0), Foragers
high-F + low-σ, Nurses low-F + low-σ** — WITH the principled entropy leak and no force cap. The coupling lesson
(force-flow ↔ threshold-response) still holds; the fix is to model the missing spatial coupling, not to decouple.
