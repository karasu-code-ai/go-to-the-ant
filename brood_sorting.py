#!/usr/bin/env python3
r"""'Go to the Ant' §3.2 — Ant Brood Sorting (Deneubourg et al. 1991), recreated from the paper.

An ant hill keeps larvae, eggs, cocoons, food sorted by kind — but no ant runs a sorting algorithm.
Parunak's four local rules (§3.2), verbatim in spirit:
  1. Wander randomly around the nest.
  2. Keep a SHORT memory (~10 steps) of the object types recently seen.
  3. Not carrying + at an object: pick it up stochastically. p(pickup) = (k+/(k+ + f))^2, where f is the
     fraction of short-term memory occupied by the SAME type. (Rare type -> f small -> pick up ~surely.)
  4. Carrying + on empty ground: drop it stochastically. p(putdown) = (f/(k- + f))^2. (Surrounded by the
     same type -> f large -> drop ~surely.)
  Constants (paper): k+ ~ 1, k- ~ 3 (k- must exceed k+ or clusters dissolve faster than they form).
Local concentrations of like items emerge, retain members, and attract more; stochastic pickup lets
separate clusters merge. Sorting EMERGES; no ant compares the whole nest.

  python brood_sorting.py                 # ASCII before/after + a cluster-quality curve
"""
import argparse, random
from collections import deque

TYPES = "ABC"                      # kinds of brood items (larvae/eggs/cocoons)


class Nest:
    def __init__(self, w=40, h=24, n_per_type=90, seed=0):
        self.w, self.h = w, h
        self.rng = random.Random(seed)
        self.grid = [[None] * w for _ in range(h)]         # None or a type char
        cells = [(x, y) for y in range(h) for x in range(w)]
        self.rng.shuffle(cells)
        i = 0
        for t in TYPES:                                    # scatter each type at random
            for _ in range(n_per_type):
                x, y = cells[i]; i += 1
                self.grid[y][x] = t

    def clustering(self):
        """Mean fraction of 8-neighbours that share an item's type — 0=scattered, 1=perfectly sorted."""
        tot = same = 0
        for y in range(self.h):
            for x in range(self.w):
                t = self.grid[y][x]
                if t is None:
                    continue
                neigh = simt = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == dy == 0:
                            continue
                        nx, ny = (x + dx) % self.w, (y + dy) % self.h
                        if self.grid[ny][nx] is not None:
                            neigh += 1
                            simt += (self.grid[ny][nx] == t)
                if neigh:
                    tot += 1; same += simt / neigh
        return same / max(tot, 1)


class SortAnt:
    def __init__(self, nest, mem=10, kp=1.0, km=3.0):
        self.n = nest; self.x = nest.rng.randrange(nest.w); self.y = nest.rng.randrange(nest.h)
        self.carry = None
        self.mem = deque(maxlen=mem)                       # rule 2: short memory of seen types
        self.kp, self.km = kp, km

    def _f(self, t):
        if not self.mem:
            return 0.0
        return sum(1 for m in self.mem if m == t) / len(self.mem)

    def step(self):
        n = self.n
        # rule 1: wander
        self.x = (self.x + n.rng.choice((-1, 0, 1))) % n.w
        self.y = (self.y + n.rng.choice((-1, 0, 1))) % n.h
        here = n.grid[self.y][self.x]
        self.mem.append(here)                              # rule 2 (record even empties)
        if self.carry is None:
            if here is not None:                           # rule 3: maybe pick up
                p = (self.kp / (self.kp + self._f(here))) ** 2   # PAPER §3.2 VERBATIM: p(pickup)=(k+/(k++f))^2
                if n.rng.random() < p:
                    self.carry = here; n.grid[self.y][self.x] = None
        else:
            if here is None:                               # rule 4: maybe drop
                p = (self._f(self.carry) / (self.km + self._f(self.carry))) ** 2  # PAPER §3.2 VERBATIM: p(putdown)=(f/(k-+f))^2
                if n.rng.random() < p:
                    n.grid[self.y][self.x] = self.carry; self.carry = None


def render(nest):
    for row in nest.grid:
        print("".join(c if c else "." for c in row))


def run(ticks=120000, n_ants=40, seed=0, verbose=True):
    nest = Nest(seed=seed)
    ants = [SortAnt(nest) for _ in range(n_ants)]
    if verbose:
        print("BEFORE (random scatter):\n"); render(nest)
        print(f"\ninitial clustering: {nest.clustering():.3f}")
    hist = []
    for t in range(ticks):
        for a in ants:
            a.step()
        if t % max(1, ticks // 12) == 0:
            hist.append(nest.clustering())
    if verbose:
        print("\nAFTER (emergent sorting):\n"); render(nest)
        print(f"\nfinal clustering: {nest.clustering():.3f}")
        print("clustering(t):", " ".join(f"{c:.2f}" for c in hist))
    return nest, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=120000)
    ap.add_argument("--ants", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.ticks, a.ants, a.seed)


if __name__ == "__main__":
    main()
