#!/usr/bin/env python3
r"""'Go to the Ant' §3.5 — Birds & Fish: Flocking (Reynolds 1987, Heppner 1990), recreated from the paper.

Flocks stay together, turn together, and avoid collisions with no leader and no central coordinator — each
bird senses only its nearest peers. Parunak lists Reynolds' three local rules (§3.5), which are the paper's:
  1. SEPARATION — keep a minimum distance from the nearest birds (avoid collisions).
  2. ALIGNMENT  — match velocity (speed + heading) to nearby birds.
  3. COHESION   — stay close to the centre of the local flock.
Each rule is a steering vector from the neighbours inside a perception radius; their sum turns the bird.
Global coordination (a single coherent, banking flock) EMERGES from these three local urges.

PROVENANCE: the three rules are the paper's (Reynolds' "boids"). The perception radius, the separation
distance, and the three weights are OPERATIONALIZED — Parunak's paper lists the rules but gives no numbers
(Reynolds 1987 is the primary source for tuned constants).

  python flocking.py                 # ASCII flock + a polarization (alignment) curve
"""
import argparse, math, random


class Flock:
    def __init__(self, n=90, w=90, h=48, seed=0,
                 perc=8.0, sep_r=3.0, wsep=1.3, wali=1.5, wcoh=0.85, vmax=1.0, turn=0.35):
        self.turn = turn
        self.n, self.w, self.h = n, w, h
        rng = random.Random(seed); self.rng = rng
        self.px = [rng.uniform(0, w) for _ in range(n)]
        self.py = [rng.uniform(0, h) for _ in range(n)]
        ang = [rng.uniform(0, 2 * math.pi) for _ in range(n)]
        self.vx = [math.cos(a) for a in ang]; self.vy = [math.sin(a) for a in ang]
        self.perc, self.sep_r = perc, sep_r
        self.wsep, self.wali, self.wcoh, self.vmax = wsep, wali, wcoh, vmax

    def step(self):
        px, py, vx, vy, n = self.px, self.py, self.vx, self.vy, self.n
        nvx, nvy = vx[:], vy[:]
        p2, s2 = self.perc ** 2, self.sep_r ** 2
        for i in range(n):
            sx = sy = ax = ay = cx = cy = 0.0; cnt = 0
            for j in range(n):
                if i == j:
                    continue
                dx, dy = px[j] - px[i], py[j] - py[i]
                dx -= self.w * round(dx / self.w); dy -= self.h * round(dy / self.h)  # toroidal
                d2 = dx * dx + dy * dy
                if d2 > p2:
                    continue
                cnt += 1
                ax += vx[j]; ay += vy[j]            # rule 2: alignment (avg neighbour velocity)
                cx += dx; cy += dy                  # rule 3: cohesion (toward neighbour centre)
                if d2 < s2 and d2 > 1e-9:            # rule 1: separation (push from the close ones)
                    sx -= dx / d2; sy -= dy / d2
            if cnt:
                ax /= cnt; ay /= cnt; cx /= cnt; cy /= cnt
                # NORMALIZE each urge to a unit vector so the three weights are actually comparable
                # (otherwise the position-scale cohesion vector swamps the velocity-scale alignment one).
                def u(x, y):
                    m = math.hypot(x, y)
                    return (x / m, y / m) if m > 1e-9 else (0.0, 0.0)
                sux, suy = u(sx, sy)                       # rule 1: away from close birds
                aux, auy = u(ax - vx[i], ay - vy[i])       # rule 2: toward neighbours' heading
                cux, cuy = u(cx, cy)                       # rule 3: toward neighbours' centre
                accx = self.wsep * sux + self.wali * aux + self.wcoh * cux
                accy = self.wsep * suy + self.wali * auy + self.wcoh * cuy
                nvx[i] = vx[i] + self.turn * accx; nvy[i] = vy[i] + self.turn * accy
                sp = math.hypot(nvx[i], nvy[i]) or 1.0     # cap speed
                nvx[i] = nvx[i] / sp * self.vmax; nvy[i] = nvy[i] / sp * self.vmax
        for i in range(n):
            vx[i], vy[i] = nvx[i], nvy[i]
            px[i] = (px[i] + vx[i]) % self.w; py[i] = (py[i] + vy[i]) % self.h
        return self.polarization()

    def polarization(self):
        """Order parameter: |mean heading|, 0 = disordered, 1 = one coherent flock."""
        mx = sum(self.vx) / self.n; my = sum(self.vy) / self.n
        return math.hypot(mx, my) / self.vmax


def run(ticks=600, n=90, seed=0, verbose=True):
    fl = Flock(n=n, seed=seed)
    hist = [fl.polarization()]
    for t in range(ticks):
        p = fl.step()
        if t % max(1, ticks // 12) == 0:
            hist.append(p)
    if verbose:
        render(fl)
        print(f"\npolarization (flock alignment): {fl.polarization():.3f}  (0 = chaos, 1 = one flock)")
        print("polarization(t):", " ".join(f"{c:.2f}" for c in hist))
    return fl, hist


def render(fl):
    grid = [[" "] * fl.w for _ in range(fl.h)]
    arrow = "→↗↑↖←↙↓↘"
    for i in range(fl.n):
        x, y = int(fl.px[i]) % fl.w, int(fl.py[i]) % fl.h
        a = math.atan2(fl.vy[i], fl.vx[i]); k = int(round(a / (math.pi / 4))) % 8
        grid[y][x] = arrow[k]
    print("\nFlock (each bird points along its heading — watch them align):\n")
    for row in grid:
        print("".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--birds", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.ticks, a.birds, a.seed)


if __name__ == "__main__":
    main()
