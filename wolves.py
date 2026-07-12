#!/usr/bin/env python3
r"""'Go to the Ant' §3.6 — Wolves: Surrounding Prey (Korf 1992), recreated from the paper.

One wolf can't kill a moose; the pack must SURROUND it — with no radios and no negotiated strategy
("I'll take the north side, you take the south"). Parunak gives two local rules (§3.6):
  1. MOOSE: move to the neighbouring cell FARTHEST from the nearest wolf. (If it's faster than the wolves,
     it escapes; here it isn't.)
  2. WOLVES: move to minimise  S = d(moose) − k·d(nearest other wolf)  — i.e. get CLOSE to the moose while
     staying FAR from each other. k is a repulsion tuning constant.
With attraction (to prey) and repulsion (between wolves) balanced, the pack inevitably encircles the moose,
no communication required.

PROVENANCE: rules and the score S = d(moose) − k·d(wolf) are the paper's (Korf 1992). The speeds, k, and
the continuous-vs-hex board are OPERATIONALIZED (the paper states six wolves capture on a hex grid; here a
continuous plane, candidate-direction search).

  python wolves.py                 # ASCII pursuit + the encirclement gap over time
"""
import argparse, math, random

TAU = 2 * math.pi


class Hunt:
    def __init__(self, n_wolves=6, w=80, h=44, seed=0, vm=0.6, vw=1.0, k=1.12):
        rng = random.Random(seed); self.rng = rng
        self.w, self.h = w, h; self.vm, self.vw, self.k = vm, vw, k
        self.mx, self.my = w / 2, h / 2
        self.wolves = [(rng.uniform(0, w), rng.uniform(0, h)) for _ in range(n_wolves)]

    def _cands(self, x, y, sp):
        return [(x + sp * math.cos(a), y + sp * math.sin(a)) for a in
                (i * TAU / 24 for i in range(24))] + [(x, y)]

    def step(self):
        W = self.wolves
        # rule 1: moose flees to the candidate farthest from its nearest wolf
        best, bd = (self.mx, self.my), -1
        for cx, cy in self._cands(self.mx, self.my, self.vm):
            if not (0 <= cx < self.w and 0 <= cy < self.h):
                continue
            d = min(math.hypot(cx - wx, cy - wy) for wx, wy in W)
            if d > bd:
                bd, best = d, (cx, cy)
        self.mx, self.my = best
        # rule 2: each wolf minimises S = d(moose) - k*d(nearest OTHER wolf)
        nw = []
        for i, (wx, wy) in enumerate(W):
            best, bs = (wx, wy), 1e9
            for cx, cy in self._cands(wx, wy, self.vw):
                if not (0 <= cx < self.w and 0 <= cy < self.h):
                    continue
                dm = math.hypot(cx - self.mx, cy - self.my)
                do = min((math.hypot(cx - ox, cy - oy) for j, (ox, oy) in enumerate(W) if j != i),
                         default=0.0)
                s = dm - self.k * do
                if s < bs:
                    bs, best = s, (cx, cy)
            nw.append(best)
        self.wolves = nw
        return self.gap()

    def gap(self):
        """Largest angular gap (deg) between adjacent wolves as seen from the moose.
        360/N when evenly ringed → surrounded; near 360 when all on one side → open escape."""
        angs = sorted(math.atan2(wy - self.my, wx - self.mx) for wx, wy in self.wolves)
        if len(angs) < 2:
            return 360.0
        gaps = [(angs[(i + 1) % len(angs)] - angs[i]) % TAU for i in range(len(angs))]
        return max(gaps) * 180 / math.pi


def run(ticks=260, n_wolves=6, seed=0, verbose=True):
    hunt = Hunt(n_wolves=n_wolves, seed=seed)
    hist = [hunt.gap()]
    for t in range(ticks):
        g = hunt.step()
        if t % max(1, ticks // 12) == 0:
            hist.append(g)
    if verbose:
        render(hunt)
        md = min(math.hypot(hunt.mx - wx, hunt.my - wy) for wx, wy in hunt.wolves)
        print(f"\nlargest escape gap around the moose: {hunt.gap():.0f}°  "
              f"(evenly surrounded ≈ {360//n_wolves}°) | nearest wolf {md:.1f}")
        print("gap°(t):", " ".join(f"{g:.0f}" for g in hist))
    return hunt, hist


def render(hunt):
    grid = [[" "] * hunt.w for _ in range(hunt.h)]
    for wx, wy in hunt.wolves:
        x, y = int(wx) % hunt.w, int(wy) % hunt.h
        grid[y][x] = "W"
    grid[int(hunt.my) % hunt.h][int(hunt.mx) % hunt.w] = "M"
    print("\nThe hunt (M moose, W wolves — watch the ring close):\n")
    for row in grid:
        print("".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=260)
    ap.add_argument("--wolves", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.ticks, a.wolves, a.seed)


if __name__ == "__main__":
    main()
