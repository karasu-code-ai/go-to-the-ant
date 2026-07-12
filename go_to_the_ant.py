#!/usr/bin/env python3
r"""'Go to the Ant' — a faithful recreation of Parunak's OG foraging swarm.

Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural Multi-Agent Systems,"
Annals of Operations Research 75:69-101 (1997). 

Recreated from the paper + references ALONE — the five local ant rules (§3.1 "Ants: Path planning"),
verbatim in spirit:
  1. Avoid obstacles.
  2. Wander randomly, biased toward nearby pheromone. No pheromone -> Brownian motion (uniform over
     directions). Pheromone present -> same random walk, but the direction distribution is WEIGHTED toward
     the scent.
  3. If holding food, drop pheromone at a CONSTANT RATE while walking. (Optionally follow a nest beacon;
     "both approaches yield the same global behavior. The homing beacon generates paths sooner.")
  4. If at food and not holding any, pick it up.
  5. If at the nest and carrying food, drop it.
Plus the field law: pheromone EVAPORATES every tick, so paths to depleted sources — and paths laid by ants
who never got home — fade. No ant plans a route; the network (a minimum spanning tree, Goss et al. 1989)
EMERGES from deposit + evaporation + weighted-random following.

The paper's design principles this honors: agents small (a few local rules, short reach); decentralized;
share information THROUGH THE ENVIRONMENT (the pheromone field is the only channel); support entropy (the
random walk never stops, even on a trail — that wandering cuts the short-cuts across initial meanders);
pursue no optima (no ant computes a shortest path).

  python go_to_the_ant.py                       # ASCII demo (nest <-> food, wall with a gap)
  python go_to_the_ant.py --ticks 4000 --ants 120 --png trail.png
"""
import argparse, math, random


class World:
    def __init__(self, w=56, h=28, seed=0, use_wall=False, region=2):
        self.w, self.h = w, h
        self.rng = random.Random(seed)
        # PRINCIPLE APPLIED (§4.3.3 Small in Scope + §4.6 multi-marker): TWO local pheromone fields, not a
        # global homing beacon. Searchers lay a HOME trail + follow FOOD scent; carriers lay a FOOD trail +
        # follow HOME scent. Every ant now senses only local fields — no ant knows where the nest is.
        self.food_pher = [[0.0] * w for _ in range(h)]     # laid by carriers; followed by searchers
        self.home_pher = [[0.0] * w for _ in range(h)]     # laid by searchers; followed by carriers
        self.obstacle = [[False] * w for _ in range(h)]
        self.nest = (5, h // 2)
        self.food = (w - 6, h // 2)
        self.region = region                                # nest/food are small REGIONS, not points
        self.food_qty = 10 ** 9                              # effectively unlimited source
        self.gap = None
        if use_wall:                                        # wall with a single gap -> the swarm must ROUTE
            wallx = w // 2
            self.gap = self.rng.randint(4, h - 5)
            for y in range(h):
                if abs(y - self.gap) > 2:
                    self.obstacle[y][wallx] = True
        self.deliveries = 0

    def free(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h and not self.obstacle[y][x]

    def at_food(self, x, y):
        return abs(x - self.food[0]) <= self.region and abs(y - self.food[1]) <= self.region

    def at_nest(self, x, y):
        return abs(x - self.nest[0]) <= self.region and abs(y - self.nest[1]) <= self.region

    def evaporate(self, rate):
        keep = 1.0 - rate
        for fr, hr in zip(self.food_pher, self.home_pher):     # both fields dissipate (the entropy leak)
            for x in range(self.w):
                fr[x] *= keep; hr[x] *= keep

    def emit_and_diffuse_home(self):
        """The NEST is a home-pheromone SOURCE; the marker DIFFUSES outward (Brownian, §4.6) into a gradient
        that points home from everywhere. Carriers read only the LOCAL gradient — no global nest-direction."""
        nx, ny = self.nest
        for dy in range(-self.region, self.region + 1):
            for dx in range(-self.region, self.region + 1):
                x, y = nx + dx, ny + dy
                if self.free(x, y):
                    self.home_pher[y][x] += 6.0
        nxt = [row[:] for row in self.home_pher]
        for y in range(self.h):
            for x in range(self.w):
                if not self.free(x, y):
                    continue
                s = self.home_pher[y][x]; c = 1
                for dx, dy in DIRS:
                    xx, yy = x + dx, y + dy
                    if self.free(xx, yy):
                        s += self.home_pher[yy][xx]; c += 1
                nxt[y][x] = s / c
        self.home_pher = nxt


DIRS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


class Ant:
    def __init__(self, world):
        self.w = world
        self.x, self.y = world.nest
        self.carrying = False

    def _weights(self):
        """Rule 2, fully LOCAL: follow the field that leads where you're going. Searchers read the FOOD
        scent (toward food); carriers read the HOME scent (toward the nest). Brownian floor keeps the walk
        alive. No ant knows where the nest is — it just climbs the local home gradient."""
        field = self.w.home_pher if self.carrying else self.w.food_pher
        wts = []
        for dx, dy in DIRS:
            nx, ny = self.x + dx, self.y + dy
            if not self.w.free(nx, ny):                     # rule 1: never step into a wall
                wts.append(0.0); continue
            wts.append(1.0 + field[ny][nx] * 6.0)           # Brownian floor + local scent bias
        return wts

    def step(self, deposit_amt):
        wts = self._weights()
        tot = sum(wts)
        if tot <= 0:                                        # boxed in — stay put this tick
            return
        r = self.w.rng.random() * tot
        acc = 0.0
        for (dx, dy), wt in zip(DIRS, wts):
            acc += wt
            if r <= acc:
                self.x += dx; self.y += dy
                break
        # rule 3: carriers lay the FOOD trail (the nest broadcasts the HOME field, so searchers lay nothing)
        if self.carrying:
            self.w.food_pher[self.y][self.x] += deposit_amt
        # rule 4: pick up food
        if self.w.at_food(self.x, self.y) and not self.carrying and self.w.food_qty > 0:
            self.carrying = True; self.w.food_qty -= 1
        # rule 5: drop food at the nest
        elif self.w.at_nest(self.x, self.y) and self.carrying:
            self.carrying = False; self.w.deliveries += 1


def run(ticks=3000, n_ants=90, evap=0.015, deposit=1.0, seed=0, use_wall=False, verbose=True):
    world = World(seed=seed, use_wall=use_wall)
    ants = [Ant(world) for _ in range(n_ants)]
    history = []
    for t in range(ticks):
        world.emit_and_diffuse_home()                       # nest broadcasts the home gradient
        for a in ants:
            a.step(deposit)
        world.evaporate(evap)
        if t % max(1, ticks // 20) == 0:
            history.append((t, world.deliveries))
    if verbose:
        render_ascii(world, ants)
        print(f"\nfood delivered to nest over {ticks} ticks: {world.deliveries}")
        print("deliveries(t):", " ".join(f"{d}" for _, d in history))
    return world, history


def render_ascii(world, ants):
    peak = max((max(row) for row in world.food_pher), default=0.0) or 1.0
    shades = " .:-=+*#%@"
    antpos = {(a.x, a.y) for a in ants}
    print(f"\nGo to the Ant — emergent trail (nest N <-> food F, wall '|', pheromone density):\n")
    for y in range(world.h):
        line = []
        for x in range(world.w):
            if (x, y) == world.nest: c = "N"
            elif (x, y) == world.food: c = "F"
            elif world.obstacle[y][x]: c = "|"
            elif (x, y) in antpos: c = "o"
            else:
                lvl = int((world.food_pher[y][x] / peak) * (len(shades) - 1))
                c = shades[max(0, min(len(shades) - 1, lvl))]
            line.append(c)
        print("".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--ants", type=int, default=90)
    ap.add_argument("--evap", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wall", action="store_true", help="add a wall with a gap (the routing demo)")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    world, hist = run(a.ticks, a.ants, a.evap, seed=a.seed, use_wall=a.wall)
    if a.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 4.5))
            plt.imshow(world.food_pher, cmap="magma", origin="upper")
            plt.scatter([world.nest[0]], [world.nest[1]], c="cyan", s=60, marker="s", label="nest")
            plt.scatter([world.food[0]], [world.food[1]], c="lime", s=60, marker="*", label="food")
            plt.title("'Go to the Ant' (Parunak 1997) — emergent foraging trail"); plt.legend()
            plt.savefig(a.png, dpi=110, bbox_inches="tight")
            print(f"saved {a.png}")
        except Exception as e:
            print("png skipped:", e)


if __name__ == "__main__":
    main()
