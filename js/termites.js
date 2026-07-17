#!/usr/bin/env node
/*
 * "Go to the Ant" §3.3 — Termite Nest Building (Kugler/Turvey 1990), recreated
 * from the paper.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
 *
 * Tropical termites raise metre-high mounds — columns, arches, floors — with no
 * chief engineer. Parunak's three local rules (§3.3):
 *   1. Metabolize bodily waste, which contains pheromone. The waste IS the
 *      building material.
 *   2. Wander randomly, but prefer the direction of strongest local pheromone.
 *   3. Each step, decide stochastically whether to deposit the current load.
 *      p(deposit) rises with the LOCAL pheromone density AND the amount carried.
 *      A full termite drops even with no nearby deposit; a termite in a very high
 *      local concentration drops even a small load.
 * Because pheromone DECAYS, the freshest deposits (the centre of a growing pile)
 * smell strongest, so piles climb into COLUMNS rather than spreading. No termite
 * plans the mound. Emergence = scattered dabs self-concentrate into a handful of
 * tall columns.
 *
 * TWO local fields (top-down 2D): mass = persistent structural mass (what you
 * see); scent = decaying pheromone (what biases wandering).
 *
 * LENS (JavaScript): zero-dependency, standard-library only. Runs under node and
 * is portable to the browser visualizer — the web-native, accessible lens. All
 * 64-bit PRNG arithmetic uses BigInt (JS numbers are IEEE doubles); the field
 * math uses ordinary doubles, exactly as the Python reference does.
 */

'use strict';

// The 8 neighbours, in THIS exact order (matters for the weighted-random pick
// AND for the toroidal local-maximum test in columns()).
const DIRS = [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];

// ---- PRNG: SplitMix64, identical across all ports (BigInt for u64 wrapping) ----
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
  // randrange(n) -> int in [0, n)
  randrange(n) { return Number(this.next() % BigInt(n)); }
}

class Mound {
  constructor(w = 58, h = 34, seed = 0, decay = 0.02, D = 0.010) {
    this.w = w; this.h = h;
    this.rng = new SplitMix64(seed);
    this.mass = Array.from({ length: h }, () => new Float64Array(w));  // persistent structure (viz)
    this.scent = Array.from({ length: h }, () => new Float64Array(w)); // dissipative pheromone: diffuses AND decays
    this.buf = Array.from({ length: h }, () => new Float64Array(w));   // double-buffer for the diffusion pass
    this.decay = decay;
    this.D = D;                                                        // diffusion rate (Brownian spreading, §4.6)
  }

  // Field law (§4.6 entropy leak): scent DIFFUSES (Brownian spreading, a local nearest-
  // neighbour stencil) then EVAPORATES. Spreading gives each pile breadth — the substrate
  // for column skirts and inter-column arches. Double-buffered so every cell reads the OLD
  // scent; draws NO rng, so cross-port bit-identity is preserved.
  //   new = old + D*(mean8 - old);  then *= (1 - decay).
  fieldStep() {
    const keep = 1 - this.decay, D = this.D, w = this.w, h = this.h;
    for (let y = 0; y < h; y++) {
      const brow = this.buf[y];
      for (let x = 0; x < w; x++) {
        let acc = 0.0;
        for (const [dx, dy] of DIRS) {
          const nx = ((x + dx) % w + w) % w;
          const ny = ((y + dy) % h + h) % h;
          acc += this.scent[ny][nx];
        }
        const c = this.scent[y][x];
        brow[x] = (c + D * (acc / 8.0 - c)) * keep;
      }
    }
    const t = this.scent; this.scent = this.buf; this.buf = t;
  }

  // Count distinct COLUMNS = toroidal local maxima with mass above a fraction of
  // the tallest peak.
  columns() {
    let peak = 0.0;
    for (let y = 0; y < this.h; y++)
      for (let x = 0; x < this.w; x++)
        if (this.mass[y][x] > peak) peak = this.mass[y][x];
    if (peak <= 0) return [0, 0.0];
    const cut = peak * 0.15;
    let cnt = 0;
    for (let y = 0; y < this.h; y++) {
      for (let x = 0; x < this.w; x++) {
        const v = this.mass[y][x];
        if (v < cut) continue;
        let isMax = true;
        for (const [dx, dy] of DIRS) {
          const nx = ((x + dx) % this.w + this.w) % this.w;
          const ny = ((y + dy) % this.h + this.h) % this.h;
          if (v < this.mass[ny][nx]) { isMax = false; break; }
        }
        if (isMax) cnt += 1;
      }
    }
    return [cnt, peak];
  }
}

class Termite {
  constructor(m, metab = 0.4, maxload = 6.0) {
    this.m = m;
    // RNG order matches Python: randrange(w) then randrange(h), per termite.
    this.x = m.rng.randrange(m.w);
    this.y = m.rng.randrange(m.h);
    this.load = 0.0; this.metab = metab; this.maxload = maxload;
  }

  step() {
    const m = this.m;
    // rule 1: metabolize -> waste accumulates
    this.load = Math.min(this.maxload, this.load + this.metab);
    // rule 2: wander, biased toward the strongest local scent
    const wts = new Array(8);
    let tot = 0.0;
    for (let i = 0; i < 8; i++) {
      const nx = ((this.x + DIRS[i][0]) % m.w + m.w) % m.w;
      const ny = ((this.y + DIRS[i][1]) % m.h + m.h) % m.h;
      const w = 1.0 + m.scent[ny][nx] * 3.0;
      wts[i] = [w, nx, ny];
      tot += w;
    }
    let r = m.rng.random() * tot;
    for (let i = 0; i < 8; i++) {
      r -= wts[i][0];
      if (r <= 0) { this.x = wts[i][1]; this.y = wts[i][2]; break; }
    }
    // rule 3: stochastic deposit — rises with local scent AND load; full termite
    // always drops.
    const local = m.scent[this.y][this.x];
    // OPERATIONALIZED: paper §3.3 gives NO formula, only "prob rises with local
    // density AND load". This deposit-probability formula operationalizes that.
    const p = Math.min(1.0, 0.01 + 0.55 * (this.load / this.maxload) + 0.20 * local);
    // NB: short-circuit matches Python — when load>=maxload, random() is NOT
    // consumed, so the RNG stream stays bit-identical across ports.
    if (this.load >= this.maxload || m.rng.random() < p) {
      m.mass[this.y][this.x] += this.load;
      m.scent[this.y][this.x] += this.load;
      this.load = 0.0;
    }
  }
}

function run(ticks, n, seed, decay, verbose) {
  const mound = new Mound(58, 34, seed, decay);
  const termites = Array.from({ length: n }, () => new Termite(mound));
  const hist = [];
  const stride = Math.max(1, Math.floor(ticks / 12));
  for (let t = 0; t < ticks; t++) {
    for (const tm of termites) tm.step();
    mound.fieldStep();
    if (t % stride === 0) hist.push(mound.columns()[0]);
  }
  if (verbose) {
    render(mound);
    const [cnt, peak] = mound.columns();
    console.log(`\ndistinct columns (local maxima): ${cnt} | tallest column mass: ${peak.toFixed(0)}`);
    console.log('columns(t): ' + hist.join(' '));
  }
  return { mound, hist };
}

function render(mound) {
  let peak = 0.0;
  for (let y = 0; y < mound.h; y++)
    for (let x = 0; x < mound.w; x++)
      if (mound.mass[y][x] > peak) peak = mound.mass[y][x];
  if (peak === 0.0) peak = 1.0;
  const shades = ' .:-=+*#%@';
  console.log('\nTermite mound (top-down mass density — columns emerge as bright cores):\n');
  for (let y = 0; y < mound.h; y++) {
    let line = '';
    for (let x = 0; x < mound.w; x++) {
      let lvl = Math.floor((mound.mass[y][x] / peak) * (shades.length - 1));
      lvl = Math.max(0, Math.min(shades.length - 1, lvl));
      line += shades[lvl];
    }
    console.log(line);
  }
}

function main() {
  const args = process.argv.slice(2);
  const opts = { ticks: 40000, termites: 70, decay: 0.02, seed: 0 };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--ticks') opts.ticks = parseInt(args[++i], 10);
    else if (a === '--termites' || a === '--ants') opts.termites = parseInt(args[++i], 10);
    else if (a === '--decay') opts.decay = parseFloat(args[++i]);
    else if (a === '--seed') opts.seed = parseInt(args[++i], 10);
  }
  run(opts.ticks, opts.termites, opts.seed, opts.decay, true);
}

main();
