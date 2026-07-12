#!/usr/bin/env node
/*
 * "Go to the Ant" §3.5 — Birds & Fish: Flocking (Reynolds 1987, Heppner 1990),
 * a faithful port of the Python reference.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
 *
 * Flocks stay together, turn together, and avoid collisions with no leader and
 * no central coordinator — each bird senses only its nearest peers. Parunak
 * lists Reynolds' three local rules (§3.5), which are the paper's:
 *   1. SEPARATION — keep a minimum distance from the nearest birds.
 *   2. ALIGNMENT  — match velocity (speed + heading) to nearby birds.
 *   3. COHESION   — stay close to the centre of the local flock.
 * Each rule is a steering vector from the neighbours inside a perception radius;
 * their sum turns the bird. A single coherent, banking flock EMERGES from these
 * three local urges.
 *
 * PROVENANCE: the three rules are the paper's (Reynolds' "boids"). The
 * perception radius, the separation distance, and the three weights are
 * OPERATIONALIZED — Parunak's paper lists the rules but gives no numbers
 * (Reynolds 1987 is the primary source for tuned constants).
 *
 * LENS (JavaScript): zero-dependency, standard-library only. Runs under node and
 * is portable to the browser visualizer — the web-native, accessible lens. All
 * 64-bit PRNG arithmetic uses BigInt (JS numbers are IEEE doubles); the field
 * math uses ordinary doubles, exactly as the Python reference does.
 *
 * NO per-step randomness — fully deterministic given the random init, so
 * cross-port identity depends only on matching the init-RNG order and the
 * neighbour-sum order (iterate j in index order).
 */

'use strict';

// ---- PRNG: SplitMix64, identical across all ports (BigInt for u64 wrapping) ----
// CROSS-PORT CONVENTION: this intentionally does NOT reproduce CPython's random
// module — the six sequential ports agree with EACH OTHER, not with Python.
const MASK64 = (1n << 64n) - 1n;
class SplitMix64 {
  constructor(seed) { this.state = BigInt(seed) & MASK64; }
  next() {
    this.state = (this.state + 0x9E3779B97F4A7C15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }
  // (next() >> 11) * 2^-53  ->  a double in [0, 1)
  random() { return Number(this.next() >> 11n) * (1.0 / 9007199254740992.0); }
  // uniform(a, b): a + random()*(b-a)
  uniform(a, b) { return a + this.random() * (b - a); }
}

class Flock {
  constructor(n = 90, w = 90, h = 48, seed = 0,
              perc = 8.0, sep_r = 3.0, wsep = 1.3, wali = 1.5, wcoh = 0.85,
              vmax = 1.0, turn = 0.35) {
    this.turn = turn;
    this.n = n; this.w = w; this.h = h;
    this.rng = new SplitMix64(seed);
    // RNG ORDER (must match the Python reference exactly): all px, then all py,
    // then all headings. Init px,py uniform in the box; heading uniform in [0,2pi).
    this.px = new Array(n);
    this.py = new Array(n);
    for (let i = 0; i < n; i++) this.px[i] = this.rng.uniform(0, w);
    for (let i = 0; i < n; i++) this.py[i] = this.rng.uniform(0, h);
    this.vx = new Array(n);
    this.vy = new Array(n);
    for (let i = 0; i < n; i++) {
      const a = this.rng.uniform(0, 2 * Math.PI);
      this.vx[i] = Math.cos(a); this.vy[i] = Math.sin(a);
    }
    this.perc = perc; this.sep_r = sep_r;
    this.wsep = wsep; this.wali = wali; this.wcoh = wcoh; this.vmax = vmax;
  }

  step() {
    const px = this.px, py = this.py, vx = this.vx, vy = this.vy, n = this.n;
    const nvx = vx.slice(), nvy = vy.slice();
    const p2 = this.perc * this.perc, s2 = this.sep_r * this.sep_r;
    for (let i = 0; i < n; i++) {
      let sx = 0, sy = 0, ax = 0, ay = 0, cx = 0, cy = 0, cnt = 0;
      for (let j = 0; j < n; j++) {          // iterate j in index order (sum order matters)
        if (i === j) continue;
        let dx = px[j] - px[i], dy = py[j] - py[i];
        dx -= this.w * Math.round(dx / this.w);   // toroidal delta
        dy -= this.h * Math.round(dy / this.h);
        const d2 = dx * dx + dy * dy;
        if (d2 > p2) continue;
        cnt += 1;
        ax += vx[j]; ay += vy[j];            // rule 2: alignment (avg neighbour velocity)
        cx += dx; cy += dy;                  // rule 3: cohesion (toward neighbour centre)
        if (d2 < s2 && d2 > 1e-9) {          // rule 1: separation (push from the close ones)
          sx -= dx / d2; sy -= dy / d2;
        }
      }
      if (cnt) {
        ax /= cnt; ay /= cnt; cx /= cnt; cy /= cnt;
        // NORMALIZE each urge to a unit vector so the three weights are actually
        // comparable (otherwise the position-scale cohesion vector swamps the
        // velocity-scale alignment one).
        const [sux, suy] = unit(sx, sy);              // rule 1: away from close birds
        const [aux, auy] = unit(ax - vx[i], ay - vy[i]); // rule 2: toward neighbours' heading
        const [cux, cuy] = unit(cx, cy);              // rule 3: toward neighbours' centre
        const accx = this.wsep * sux + this.wali * aux + this.wcoh * cux;
        const accy = this.wsep * suy + this.wali * auy + this.wcoh * cuy;
        nvx[i] = vx[i] + this.turn * accx;
        nvy[i] = vy[i] + this.turn * accy;
        const sp = Math.hypot(nvx[i], nvy[i]) || 1.0; // cap speed
        nvx[i] = nvx[i] / sp * this.vmax;
        nvy[i] = nvy[i] / sp * this.vmax;
      }
    }
    for (let i = 0; i < n; i++) {
      vx[i] = nvx[i]; vy[i] = nvy[i];
      px[i] = mod(px[i] + vx[i], this.w);
      py[i] = mod(py[i] + vy[i], this.h);
    }
    return this.polarization();
  }

  // Order parameter: |mean heading|, 0 = disordered, 1 = one coherent flock.
  polarization() {
    let mx = 0, my = 0;
    for (let i = 0; i < this.n; i++) { mx += this.vx[i]; my += this.vy[i]; }
    mx /= this.n; my /= this.n;
    return Math.hypot(mx, my) / this.vmax;
  }
}

function unit(x, y) {
  const m = Math.hypot(x, y);
  return m > 1e-9 ? [x / m, y / m] : [0.0, 0.0];
}

// Python-style modulo (result has the sign of the divisor, always >= 0 here).
function mod(a, m) {
  return ((a % m) + m) % m;
}

function run(ticks, n, seed, verbose) {
  const fl = new Flock(n, 90, 48, seed);
  const hist = [fl.polarization()];
  const stride = Math.max(1, Math.floor(ticks / 12));
  for (let t = 0; t < ticks; t++) {
    const p = fl.step();
    if (t % stride === 0) hist.push(p);
  }
  if (verbose) {
    render(fl);
    console.log(`\npolarization (flock alignment): ${fl.polarization().toFixed(3)}  (0 = chaos, 1 = one flock)`);
    console.log('polarization(t): ' + hist.map((c) => c.toFixed(2)).join(' '));
  }
  return { fl, hist };
}

function render(fl) {
  const grid = Array.from({ length: fl.h }, () => new Array(fl.w).fill(' '));
  const arrow = '→↗↑↖←↙↓↘'; // → ↗ ↑ ↖ ← ↙ ↓ ↘
  for (let i = 0; i < fl.n; i++) {
    const x = mod(Math.floor(fl.px[i]), fl.w);
    const y = mod(Math.floor(fl.py[i]), fl.h);
    const a = Math.atan2(fl.vy[i], fl.vx[i]);
    const k = mod(Math.round(a / (Math.PI / 4)), 8);
    grid[y][x] = arrow[k];
  }
  console.log('\nFlock (each bird points along its heading — watch them align):\n');
  for (const row of grid) console.log(row.join(''));
}

function main() {
  const args = process.argv.slice(2);
  const opts = { ticks: 600, birds: 90, seed: 0 };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--ticks') opts.ticks = parseInt(args[++i], 10);
    else if (a === '--birds' || a === '--ants') opts.birds = parseInt(args[++i], 10);
    else if (a === '--seed') opts.seed = parseInt(args[++i], 10);
  }
  run(opts.ticks, opts.birds, opts.seed, true);
}

main();
