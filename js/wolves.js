#!/usr/bin/env node
/*
 * "Go to the Ant" §3.6 — Wolves: Surrounding Prey (Korf 1992), a faithful port
 * of the Python reference.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
 *
 * One wolf can't kill a moose; the pack must SURROUND it — with no radios and no
 * negotiated strategy. Parunak gives two local rules (§3.6):
 *   1. MOOSE: move to the neighbouring cell FARTHEST from the nearest wolf.
 *   2. WOLVES: move to minimise  S = d(moose) - k*d(nearest other wolf) — get
 *      CLOSE to the moose while staying FAR from each other. k is a repulsion
 *      tuning constant. With attraction (to prey) balanced against repulsion
 *      (between wolves), the pack inevitably encircles the moose, no
 *      communication required.
 *
 * PROVENANCE:
 *   VERBATIM  — the rules and the score S = d(moose) - k*d(wolf) are the paper's
 *               (Korf 1992).
 *   OPERATIONALIZED — the speeds, k=1.12, and the continuous-vs-hex board. The
 *               paper states six wolves capture on a hex grid; here we use a
 *               continuous plane with a 24-candidate-direction search (plus
 *               staying put). No per-step randomness: fully deterministic given
 *               the random init.
 *
 * LENS (JavaScript): zero-dependency, standard-library only. Runs under node and
 * is portable straight into the browser visualizer — the web-native, accessible
 * lens. All 64-bit PRNG arithmetic uses BigInt (JS numbers are IEEE doubles);
 * the field math uses ordinary doubles, exactly as the Python reference does.
 *
 * Cross-port identity depends only on matching the init-RNG order (per wolf:
 * uniform(0,w) then uniform(0,h), in wolf-index order) and the deterministic
 * candidate/neighbour iteration order.
 */

'use strict';

const TAU = 2 * Math.PI;

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

class Hunt {
  constructor(nWolves = 6, w = 80, h = 44, seed = 0, vm = 0.6, vw = 1.0, k = 1.12) {
    this.w = w; this.h = h; this.vm = vm; this.vw = vw; this.k = k;
    this.rng = new SplitMix64(seed);
    this.mx = w / 2; this.my = h / 2;
    // RNG ORDER (must match the Python reference exactly): per wolf, uniform(0,w)
    // then uniform(0,h), iterating wolves in index order.
    this.wolves = [];
    for (let i = 0; i < nWolves; i++) {
      const x = this.rng.uniform(0, w);
      const y = this.rng.uniform(0, h);
      this.wolves.push([x, y]);
    }
  }

  // The 24 direction candidates plus staying put, in this exact order.
  _cands(x, y, sp) {
    const out = [];
    for (let i = 0; i < 24; i++) {
      const a = i * TAU / 24;
      out.push([x + sp * Math.cos(a), y + sp * Math.sin(a)]);
    }
    out.push([x, y]);
    return out;
  }

  step() {
    const W = this.wolves;
    // rule 1 (VERBATIM): the moose flees to the in-bounds candidate FARTHEST
    // from its nearest wolf (maximise the min distance to any wolf).
    let best = [this.mx, this.my], bd = -1;
    for (const [cx, cy] of this._cands(this.mx, this.my, this.vm)) {
      if (!(cx >= 0 && cx < this.w && cy >= 0 && cy < this.h)) continue;
      let d = Infinity;
      for (const [wx, wy] of W) {
        const dd = Math.hypot(cx - wx, cy - wy);
        if (dd < d) d = dd;
      }
      if (d > bd) { bd = d; best = [cx, cy]; }
    }
    this.mx = best[0]; this.my = best[1];
    // rule 2 (VERBATIM score): each wolf minimises S = d(moose) - k*d(nearest
    // OTHER wolf). All wolves read the SAME current W (positions updated after
    // the pass), matching the Python reference.
    const nw = [];
    for (let i = 0; i < W.length; i++) {
      const wx = W[i][0], wy = W[i][1];
      let bestW = [wx, wy], bs = 1e9;
      for (const [cx, cy] of this._cands(wx, wy, this.vw)) {
        if (!(cx >= 0 && cx < this.w && cy >= 0 && cy < this.h)) continue;
        const dm = Math.hypot(cx - this.mx, cy - this.my);
        let doo = Infinity, seen = false;
        for (let j = 0; j < W.length; j++) {
          if (j === i) continue;
          seen = true;
          const dd = Math.hypot(cx - W[j][0], cy - W[j][1]);
          if (dd < doo) doo = dd;
        }
        if (!seen) doo = 0.0;
        const s = dm - this.k * doo;
        if (s < bs) { bs = s; bestW = [cx, cy]; }
      }
      nw.push(bestW);
    }
    this.wolves = nw;
    return this.gap();
  }

  // Largest angular gap (deg) between adjacent wolves as seen from the moose.
  // 360/N when evenly ringed -> surrounded; near 360 when all on one side.
  gap() {
    const angs = this.wolves.map(([wx, wy]) => Math.atan2(wy - this.my, wx - this.mx));
    angs.sort((a, b) => a - b);
    if (angs.length < 2) return 360.0;
    let mg = 0;
    for (let i = 0; i < angs.length; i++) {
      let g = angs[(i + 1) % angs.length] - angs[i];
      g = ((g % TAU) + TAU) % TAU;
      if (g > mg) mg = g;
    }
    return mg * 180 / Math.PI;
  }

  nearestWolf() {
    let md = Infinity;
    for (const [wx, wy] of this.wolves) {
      const d = Math.hypot(this.mx - wx, this.my - wy);
      if (d < md) md = d;
    }
    return md;
  }
}

function run(ticks, nWolves, seed, verbose) {
  const hunt = new Hunt(nWolves, 80, 44, seed);
  const hist = [hunt.gap()];
  const stride = Math.max(1, Math.floor(ticks / 12));
  for (let t = 0; t < ticks; t++) {
    const g = hunt.step();
    if (t % stride === 0) hist.push(g);
  }
  if (verbose) {
    render(hunt);
    const md = hunt.nearestWolf();
    const even = Math.floor(360 / nWolves);
    console.log(`\nlargest escape gap around the moose: ${hunt.gap().toFixed(0)}°  ` +
      `(evenly surrounded ≈ ${even}°) | nearest wolf ${md.toFixed(1)}`);
    console.log('gap°(t): ' + hist.map((g) => g.toFixed(0)).join(' '));
  }
  return { hunt, hist };
}

function render(hunt) {
  const grid = Array.from({ length: hunt.h }, () => new Array(hunt.w).fill(' '));
  for (const [wx, wy] of hunt.wolves) {
    const x = mod(Math.floor(wx), hunt.w), y = mod(Math.floor(wy), hunt.h);
    grid[y][x] = 'W';
  }
  grid[mod(Math.floor(hunt.my), hunt.h)][mod(Math.floor(hunt.mx), hunt.w)] = 'M';
  console.log('\nThe hunt (M moose, W wolves — watch the ring close):\n');
  for (const row of grid) console.log(row.join(''));
}

// Python-style modulo (result has the sign of the divisor, always >= 0 here).
function mod(a, m) {
  return ((a % m) + m) % m;
}

function main() {
  const args = process.argv.slice(2);
  const opts = { ticks: 260, wolves: 6, seed: 0 };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--ticks') opts.ticks = parseInt(args[++i], 10);
    else if (a === '--wolves' || a === '--ants') opts.wolves = parseInt(args[++i], 10);
    else if (a === '--seed') opts.seed = parseInt(args[++i], 10);
  }
  run(opts.ticks, opts.wolves, opts.seed, true);
}

main();
