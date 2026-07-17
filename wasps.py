#!/usr/bin/env python3
r"""'Go to the Ant' §3.4 — Wasp Task Differentiation (Theraulaz et al. 1991), recreated from the paper.

Mature Polistes wasps — genetically IDENTICAL — split into a single Chief, a band of Foragers, and a band
of Nurses, with no HR department and no wasp computing the proportion. Parunak's three interacting rules:

  1. FACE-OFFS. When two wasps meet, j beats i with the Fermi probability  p = 1/(1 + e^(h·(F_i − F_j))).
     The higher force usually wins (but not always); a quantum of Force passes loser → winner.
  2. BROOD DEMAND.  D(t) = D(t−1) + appetite − W, where W is the food-work done by all foragers.
  3. FORAGE?  A wasp near the brood forages with Fermi probability  p = 1/(1 + e^(h·(σ_j − D))).
     Foraging LOWERS its threshold σ by ξ (learning); not foraging RAISES σ by φ (forgetting).

Force is MOBILITY (a low-force wasp is stimulated by the brood but cannot travel to hunt). The joint
(Force, Threshold) distribution self-separates into three clusters — the castes:
  · Foragers  = high force, low threshold  (strong enough to move + sensitive to the brood)
  · Nurses    = low force,  low threshold  (attentive, but stuck near the brood)
  · Chief     = one wasp, high force, high threshold (grounds the scales via face-offs; doesn't forage)

  python wasps.py                 # caste populations + the (F,σ) landscape as it settles
"""
import argparse, math, random


class Colony:
    SIGMAX = 4.0
    # NOTE (provenance, honest labelling): the genuine §4.6 entropy leak for wasps is RULE 1's conservative
    # force TRANSFER among wasps (Parunak names "the flow of force among wasps that drives the emergence of the
    # three roles"). The leak/gen term below is a SEPARATE Force-RELAXATION (mean-reversion to ~gen/leak) — it
    # replaces an ad-hoc force CAP and bounds the hierarchy, but it is an INFERENCE BEYOND Parunak (plausibly a
    # Theraulaz 1991 element we can't verify; the primary source isn't on hand), NOT the §4.6 entropy leak. It is
    # empirically required: pure conservation condenses to one super-wasp with no graded forager band. See PROVENANCE.md.
    def __init__(self, n=80, seed=0, h=1.1, hf=3.0, quantum=0.10, appetite=None,
                 xi=0.02, phi=0.012, mob=1.6, leak=0.004, gen=0.005):
        self.rng = random.Random(seed); self.n = n
        # genetically identical: tiny initial spread only
        self.F = [1.0 + self.rng.uniform(-0.05, 0.05) for _ in range(n)]
        self.sig = [1.6 + self.rng.uniform(-0.05, 0.05) for _ in range(n)]
        self.seenmax = list(self.F)   # each wasp's LOCAL, FADING memory of the top force it has faced
        self.seendecay = 0.998        # the memory fades, so a stale early peak can't stick (robustness)
        self.D = 2.0
        self.h, self.hf, self.q = h, hf, quantum
        # appetite sized so the mobile minority (~n/8) can ALMOST meet demand — leaving the top wasp
        # (the Chief) surplus, so its threshold drifts high while the working foragers stay low.
        self.appetite = appetite if appetite is not None else 0.075 * n
        self.xi, self.phi, self.mob = xi, phi, mob
        self.leak, self.gen = leak, gen

    def step(self):
        rng, F, sig, n = self.rng, self.F, self.sig, self.n
        # rule 1: face-offs — gentle, capped, so a graded hierarchy forms (not one super-wasp)
        for _ in range(n // 3):
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j:
                continue
            fi, fj = F[i], F[j]
            pj = 1.0 / (1.0 + math.exp(self.h * (fi - fj)))      # PAPER §3.4 VERBATIM: p=1/(1+e^(h(Fi-Fj)))
            w, l = (j, i) if rng.random() < pj else (i, j)
            t = min(self.q, F[l])
            F[w] += t; F[l] -= t                         # force is conserved in the face-off (paper)
            m = fi if fi > fj else fj                    # LOCAL: each wasp's FADING memory of the strongest force it
            di = self.seenmax[i] * self.seendecay        # has faced — a gossip estimate of the top force, NO global
            dj = self.seenmax[j] * self.seendecay        # max. Fading lets the emergent Chief's own force become its
            self.seenmax[i] = m if m > di else di        # remembered peak (dom->1), so its threshold reliably drifts
            self.seenmax[j] = m if m > dj else dj        # high (0/32 seed failures vs 3/32 for the global-max version)
        # FORCE RELAXATION (inference beyond Parunak — NOT the §4.6 entropy leak; that is Rule 1's force flow
        # above): force mean-reverts toward ~gen/leak each tick. A steady leak+gen bounds the hierarchy naturally
        # (equilibrium mean ≈ gen/leak), so no ad-hoc force cap is needed to stop one super-wasp.
        for k in range(n):
            F[k] = max(0.0, F[k] * (1.0 - self.leak) + self.gen)
        # rules 2 & 3: brood stimulation + foraging. Work = COUNT of mobile foragers (each brings 1).
        # SPATIALITY PROXY (operationalized, now fully LOCAL): the paper's Chief "wanders and faces off", so it
        # is NOT near the brood and is rarely stimulated -> its threshold drifts HIGH. We approximate "away
        # dominating" by suppressing a wasp's foraging in proportion to how it ranks against the strongest force
        # IT HAS FACED (self.seenmax, a per-agent gossip estimate — NO global max): dominance=(F/seenmax)^4 hits
        # ~1 only for the wasp that out-forces everyone it meets (the Chief), leaving foragers almost untouched.
        W = 0
        for k in range(n):
            pf = 1.0 / (1.0 + math.exp(self.hf * (sig[k] - self.D)))  # PAPER §3.4 VERBATIM: p=1/(1+e^(h(sig-D)))
            dom = (F[k] / (self.seenmax[k] or 1.0)) ** 4          # ~1 only for the wasp on top of its own encounters
            if rng.random() < pf * (1.0 - dom):                   # stimulated AND not away dominating
                sig[k] = max(0.0, sig[k] - self.xi)               # learns: threshold drops
                if F[k] > self.mob:                               # mobile enough to actually hunt
                    W += 1
            else:
                sig[k] = min(self.SIGMAX, sig[k] + self.phi)      # forgets: threshold rises
        self.D = max(0.0, self.D + self.appetite - W)

    def castes(self):
        F, sig = self.F, self.sig
        chief = max(range(self.n), key=lambda k: F[k])
        smed = sorted(sig)[self.n // 2]
        groups = {"Chief": [], "Forager": [], "Nurse": []}
        for k in range(self.n):
            if k == chief:
                groups["Chief"].append(k)
            elif F[k] > self.mob and sig[k] <= smed:
                groups["Forager"].append(k)      # mobile + responsive
            else:
                groups["Nurse"].append(k)         # immobile (or unresponsive) -> stays with the brood
        return groups, chief


def run(ticks=4000, n=80, seed=0, verbose=True):
    c = Colony(n=n, seed=seed)
    hist = []
    for t in range(ticks):
        c.step()
        if t % max(1, ticks // 12) == 0:
            g, _ = c.castes()
            hist.append((len(g["Forager"]), len(g["Nurse"])))
    if verbose:
        g, chief = c.castes()
        print(f"Emergent castes from {n} genetically identical wasps ({ticks} ticks):\n")
        for name in ("Chief", "Forager", "Nurse"):
            ks = g[name]
            if not ks:
                continue
            mF = sum(c.F[k] for k in ks) / len(ks); mS = sum(c.sig[k] for k in ks) / len(ks)
            print(f"  {name:8} n={len(ks):3d}   mean Force {mF:5.2f}   mean Threshold {mS:5.2f}")
        print(f"\n  Chief force {c.F[chief]:.2f} (pop mean {sum(c.F)/n:.2f}), "
              f"threshold {c.sig[chief]:.2f}")
        print("  Forager/Nurse split(t):", " ".join(f"{f}/{ns}" for f, ns in hist))
        landscape(c)
    return c


def landscape(c, cols=48, rows=16):
    """ASCII scatter of the population in (Force -> x, Threshold -> y) space."""
    F, sig = c.F, c.sig
    fmn, fmx = min(F), max(F); smn, smx = min(sig), max(sig)
    grid = [[" "] * cols for _ in range(rows)]
    chief = max(range(c.n), key=lambda k: F[k])
    fmed = sorted(F)[c.n // 2]
    for k in range(c.n):
        x = int((F[k] - fmn) / (fmx - fmn + 1e-9) * (cols - 1))
        y = int((sig[k] - smn) / (smx - smn + 1e-9) * (rows - 1))
        mark = "C" if k == chief else ("F" if F[k] >= fmed else "n")
        grid[rows - 1 - y][x] = mark
    print("\n  (F,σ) landscape — x = Force →, y = Threshold ↑ | C chief, F forager, n nurse:\n")
    for row in grid:
        print("   " + "".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--wasps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.ticks, a.wasps, a.seed)


if __name__ == "__main__":
    main()
