#!/usr/bin/env python3
r"""'Go to the Ant' §3.3 — Termite Nest Building (Kugler et al. 1990), recreated from the paper.

Tropical termites raise 5-metre mounds — columns, arches, floors — with no chief engineer. Parunak's three
local rules (§3.3):
  1. Metabolize bodily waste, which contains pheromone. The waste IS the building material.
  2. Wander randomly, but prefer the direction of the strongest local pheromone concentration.
  3. Each step, decide stochastically whether to deposit the current load. p(deposit) rises with the LOCAL
     pheromone density AND the amount carried. A full termite drops even with no nearby deposit; a termite in
     a very high local concentration drops even a small load.
The pheromone field is dissipative in Parunak's §4.6 sense: it both DIFFUSES (Brownian spreading, a local
nearest-neighbour stencil) AND DECAYS every tick. Spreading gives each pile some BREADTH — a graded skirt —
so the freshest deposits (the centre of a growing pile) still smell strongest and piles climb into COLUMNS,
while each column's skirt reaches a little across open space so two nearby columns each pull the other's
visitors, bending subsequent deposits into an ARCH. No termite plans the mound.

Here (top-down 2D): `mound` = persistent structural mass (what you see); `scent` = decaying pheromone (what
biases wandering). Emergence = scattered dabs self-concentrate into a handful of tall columns.

  python termites.py                 # ASCII mound + column count over time
"""
import argparse, math, random

DIRS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


class Mound:
    def __init__(self, w=58, h=34, seed=0, decay=0.02, D=0.010):
        self.w, self.h = w, h
        self.rng = random.Random(seed)
        self.mass = [[0.0] * w for _ in range(h)]          # persistent structure (viz)
        self.scent = [[0.0] * w for _ in range(h)]         # decaying pheromone (bias)
        self._buf = [[0.0] * w for _ in range(h)]          # double-buffer for the diffusion pass
        self.decay = decay
        self.D = D                                         # diffusion rate (Brownian spreading)

    def field_step(self):
        """Process_e (§4.6): the scent DIFFUSES then EVAPORATES. Spreading is a local
        nearest-neighbour stencil — physically what Brownian motion is — that turns point
        deposits into a graded field agents orient to; decay is the forgetting half. Double-
        buffered so every cell reads the OLD scent; draws NO rng (bit-identity preserved).
            new = old + D*(mean8 - old);  then *= (1 - decay)."""
        s, buf = self.scent, self._buf
        w, h, D, keep = self.w, self.h, self.D, 1 - self.decay
        for y in range(h):
            for x in range(w):
                acc = 0.0
                for dx, dy in DIRS:
                    acc += s[(y + dy) % h][(x + dx) % w]
                c = s[y][x]
                buf[y][x] = (c + D * (acc / 8.0 - c)) * keep
        self.scent, self._buf = buf, s                     # swap buffers

    def columns(self):
        """Count distinct COLUMNS = local maxima above a fraction of the tallest peak."""
        peak = max((max(r) for r in self.mass), default=0.0)
        if peak <= 0:
            return 0, 0.0
        cut = peak * 0.15
        cnt = 0
        for y in range(self.h):
            for x in range(self.w):
                v = self.mass[y][x]
                if v < cut:
                    continue
                if all(v >= self.mass[(y+dy) % self.h][(x+dx) % self.w] for dx, dy in DIRS):
                    cnt += 1
        return cnt, peak


class Termite:
    def __init__(self, m, metab=0.4, maxload=6.0):
        self.m = m; self.x = m.rng.randrange(m.w); self.y = m.rng.randrange(m.h)
        self.load = 0.0; self.metab = metab; self.maxload = maxload

    def step(self):
        m = self.m
        # rule 1: metabolize -> waste accumulates
        self.load = min(self.maxload, self.load + self.metab)
        # rule 2: wander, biased toward the strongest local scent
        wts, tot = [], 0.0
        for dx, dy in DIRS:
            nx, ny = (self.x + dx) % m.w, (self.y + dy) % m.h
            w = 1.0 + m.scent[ny][nx] * 3.0
            wts.append((w, nx, ny)); tot += w
        r = m.rng.random() * tot
        for w, nx, ny in wts:
            r -= w
            if r <= 0:
                self.x, self.y = nx, ny; break
        # rule 3: stochastic deposit — rises with local scent AND load; full termite always drops
        local = m.scent[self.y][self.x]
        p = min(1.0, 0.01 + 0.55 * (self.load / self.maxload) + 0.20 * local)  # OPERATIONALIZED: paper §3.3 gives NO formula, only "prob rises with local density AND load"
        if self.load >= self.maxload or m.rng.random() < p:
            m.mass[self.y][self.x] += self.load
            m.scent[self.y][self.x] += self.load
            self.load = 0.0


def run(ticks=40000, n=70, seed=0, decay=0.02, verbose=True):
    mound = Mound(seed=seed, decay=decay)
    termites = [Termite(mound) for _ in range(n)]
    hist = []
    for t in range(ticks):
        for tm in termites:
            tm.step()
        mound.field_step()
        if t % max(1, ticks // 12) == 0:
            hist.append(mound.columns()[0])
    if verbose:
        render(mound)
        cnt, peak = mound.columns()
        print(f"\ndistinct columns (local maxima): {cnt} | tallest column mass: {peak:.0f}")
        print("columns(t):", " ".join(str(c) for c in hist))
    return mound, hist


def render(mound):
    peak = max((max(r) for r in mound.mass), default=1.0) or 1.0
    shades = " .:-=+*#%@"
    print("\nTermite mound (top-down mass density — columns emerge as bright cores):\n")
    for row in mound.mass:
        print("".join(shades[max(0, min(len(shades) - 1, int(v / peak * (len(shades) - 1))))] for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=40000)
    ap.add_argument("--termites", type=int, default=70)
    ap.add_argument("--decay", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.ticks, a.termites, a.seed, a.decay)


if __name__ == "__main__":
    main()
