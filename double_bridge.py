#!/usr/bin/env python3
r"""'Go to the Ant' §3.1 companion — the GOSS DOUBLE BRIDGE (Goss, Aron, Deneubourg & Pasteels 1989,
"Self-organized shortcuts in the Argentine ant," Naturwissenschaften 76:579-581), recreated from the source.

This is the canonical demonstration of ant PATH OPTIMIZATION over time — the thing the open-plane forager
cannot show, because the open plane offers no choice between a longer and a shorter route. Here the nest and
the food are joined by TWO branches of different length. No ant measures anything; the colony CONVERGES on the
short branch by pure stigmergy:

  1. CHOICE. At a fork, an ant picks branch i with the paper's nonlinear rule
        p_i = (c + φ_i)^n / Σ_j (c + φ_j)^n         (c ≈ 20, n = 2)
     — an unmarked branch has baseline attraction c; the exponent n>1 amplifies small pheromone differences.
  2. DEPOSIT. Crossing a branch lays a fixed quantum of pheromone on it.
  3. DIFFERENTIAL REINFORCEMENT (the engine). An ant that took the SHORT branch reaches the far end SOONER,
     so it re-chooses and re-deposits sooner — the short branch accumulates pheromone faster per unit time.
     Positive feedback + the nonlinear choice tip the whole colony onto the short branch.

Unlike the OPEN-PLANE forager (a single homing gradient, no route choice), the bridge topology makes homing
trivial (an ant just shuttles nest↔food) and isolates the one thing that optimizes: the branch decision. That
is why Goss used it, and why the "shortest path emerges over time" claim is demonstrable here and not there.

  python double_bridge.py            # convergence of the colony onto the short branch, over time
"""
import argparse, random


class DoubleBridge:
    def __init__(self, n_ants=48, ratio=2.0, short_len=20, c=20.0, n=2, q=1.0, evap=0.005, seed=0):
        self.rng = random.Random(seed)
        self.Ls = short_len
        self.Ll = int(round(short_len * ratio))     # long branch = ratio × short (Goss used ~2:1)
        self.phi = {"S": 0.0, "L": 0.0}             # pheromone on each branch
        self.c, self.n, self.q, self.evap = c, n, q, evap
        # STAGGERED starts: real ants don't all leave the nest at once. Seeding every ant mid-traverse on a
        # random branch desynchronizes arrivals — without it, a synchronized first pulse can lock the colony
        # onto the LONG branch. (With it, the short branch wins in ~every seed, as Goss observed.)
        self.ants = [{"branch": self.rng.choice(("S", "L")), "prog": 0} for _ in range(n_ants)]
        for a in self.ants:
            a["prog"] = self.rng.randint(0, self.Ll)
        self.recent = []                            # rolling window of recent choices (1=short, 0=long)

    def _choose(self):
        """PAPER VERBATIM (Goss 1989 eq.): p_i ∝ (c + φ_i)^n."""
        s = (self.c + self.phi["S"]) ** self.n
        l = (self.c + self.phi["L"]) ** self.n
        return "S" if self.rng.random() < s / (s + l) else "L"

    def step(self):
        for a in self.ants:
            if a["branch"] is None:                 # at a node: choose a branch, start crossing
                a["branch"] = self._choose()
                a["prog"] = 0
                self.recent.append(1 if a["branch"] == "S" else 0)
                if len(self.recent) > 300:
                    self.recent.pop(0)
            a["prog"] += 1
            length = self.Ls if a["branch"] == "S" else self.Ll
            if a["prog"] >= length:                 # arrived: lay a fixed quantum, then re-choose next tick
                self.phi[a["branch"]] += self.q
                a["branch"] = None
        if self.evap > 0:                           # optional slow evaporation (Goss's basic effect needs none)
            self.phi["S"] *= (1 - self.evap)
            self.phi["L"] *= (1 - self.evap)

    def frac_short(self):
        return sum(self.recent) / len(self.recent) if self.recent else 0.5


def run(ticks=6000, n_ants=48, ratio=2.0, seed=0, verbose=True):
    b = DoubleBridge(n_ants=n_ants, ratio=ratio, seed=seed)
    hist = []
    for t in range(ticks):
        b.step()
        if t % max(1, ticks // 20) == 0:
            hist.append(b.frac_short())
    if verbose:
        print(f"Goss double bridge — {n_ants} ants, long:short = {ratio:.0f}:1 "
              f"(Ls={b.Ls}, Ll={b.Ll}), {ticks} ticks\n")
        print(f"  pheromone: short {b.phi['S']:.0f}   long {b.phi['L']:.0f}")
        print(f"  colony on the SHORT branch: {b.frac_short()*100:.0f}%  (started ~50%)")
        print("\n  % on short over time (each cell = a slice of the run):\n")
        bar = "".join("█" if f > 0.9 else "▓" if f > 0.75 else "▒" if f > 0.55 else "░" for f in hist)
        print("   50% " + bar + f" {b.frac_short()*100:.0f}%")
        print("        (░ ~half   ▒ leaning short   ▓ mostly short   █ converged)")
    return b, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--ants", type=int, default=48)
    ap.add_argument("--ratio", type=float, default=2.0, help="long:short length ratio")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.ticks, a.ants, a.ratio, a.seed)


if __name__ == "__main__":
    main()
