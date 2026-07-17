#!/usr/bin/env node
/*
 * "Go to the Ant" §3.2 — Ant Brood Sorting (Deneubourg et al. 1991), recreated
 * from the paper.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
 *
 * An ant hill keeps larvae, eggs, cocoons sorted by kind — but no ant runs a
 * sorting algorithm. Parunak's four local rules (§3.2):
 *   1. Wander randomly around the nest.
 *   2. Keep a SHORT memory (~15 steps) of the object types recently seen.
 *   3. Not carrying + at an object: pick it up stochastically.
 *        p(pickup) = (k+/(k+ + f))^2   -- PAPER §3.2 VERBATIM
 *      where f is the fraction of short-term memory holding the SAME type.
 *      (Rare type -> f small -> pick up ~surely.)
 *   4. Carrying + on empty ground: drop it stochastically.
 *        p(putdown) = (f/(k- + f))^2   -- PAPER §3.2 VERBATIM
 *      (Surrounded by the same type -> f large -> drop ~surely.)
 *   Constants (paper): k+ 0.1, k- 0.3 (Deneubourg 1991; Parunak's summary rounds to ~1, ~3) -- k- must exceed k+ or clusters dissolve
 *   faster than they form (PAPER §3.2 VERBATIM: kp=0.1 < km=0.3, mem=15).
 * Local concentrations of like items emerge, retain members, and attract more;
 * stochastic pickup lets separate clusters merge. Sorting EMERGES; no ant
 * compares the whole nest.
 *
 * LENS (JavaScript): zero-dependency, standard-library only. Runs under node and
 * is portable straight to the browser visualizer — the web-native lens. All
 * 64-bit PRNG arithmetic uses BigInt (JS numbers are IEEE doubles); the field
 * math uses ordinary doubles, exactly as the Python reference does.
 */

'use strict';

const TYPES = 'ABC';               // kinds of brood items (larvae/eggs/cocoons)

// ---- PRNG: SplitMix64, identical across all ports (BigInt for u64 wrapping) ----
// CROSS-PORT CONVENTION: this intentionally does NOT reproduce CPython's random
// module; the goal is that all six ports agree with EACH OTHER, consuming the
// RNG in the same order. Mirrors forage.js exactly.
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
  randrange(n) { return Number(this.next() % BigInt(n)); }             // int in [0, n)
  choice(list) { return list[this.randrange(list.length)]; }
  // Fisher-Yates, matching Python's random.shuffle ordering exactly.
  shuffle(arr) {
    for (let i = arr.length - 1; i >= 1; i--) {
      const j = this.randrange(i + 1);
      const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
  }
}

class Nest {
  constructor(w = 40, h = 24, nPerType = 90, seed = 0) {
    this.w = w; this.h = h;
    this.rng = new SplitMix64(seed);
    this.grid = Array.from({ length: h }, () => new Array(w).fill(null)); // null or type char
    const cells = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) cells.push([x, y]);
    this.rng.shuffle(cells);
    let i = 0;
    for (const t of TYPES) {                          // scatter each type at random
      for (let k = 0; k < nPerType; k++) {
        const [x, y] = cells[i]; i += 1;
        this.grid[y][x] = t;
      }
    }
  }

  // Mean fraction of 8-neighbours that share an item's type — 0=scattered, 1=sorted.
  clustering() {
    let tot = 0, same = 0.0;
    for (let y = 0; y < this.h; y++) {
      for (let x = 0; x < this.w; x++) {
        const t = this.grid[y][x];
        if (t === null) continue;
        let neigh = 0, simt = 0;
        for (let dx = -1; dx <= 1; dx++) {
          for (let dy = -1; dy <= 1; dy++) {
            if (dx === 0 && dy === 0) continue;
            const nx = (x + dx + this.w) % this.w, ny = (y + dy + this.h) % this.h;
            if (this.grid[ny][nx] !== null) {
              neigh += 1;
              if (this.grid[ny][nx] === t) simt += 1;
            }
          }
        }
        if (neigh) { tot += 1; same += simt / neigh; }
      }
    }
    return same / Math.max(tot, 1);
  }
}

class SortAnt {
  constructor(nest, mem = 15, kp = 0.1, km = 0.3) {
    this.n = nest;
    this.x = nest.rng.randrange(nest.w);
    this.y = nest.rng.randrange(nest.h);
    this.carry = null;
    this.memCap = mem;                                // rule 2: short memory of seen types
    this.mem = [];
    this.kp = kp; this.km = km;
  }

  _push(v) {                                          // deque(maxlen=mem) semantics
    this.mem.push(v);
    if (this.mem.length > this.memCap) this.mem.shift();
  }

  _f(t) {
    if (this.mem.length === 0) return 0.0;
    let c = 0;
    for (const m of this.mem) if (m === t) c += 1;
    return c / this.mem.length;
  }

  step() {
    const n = this.n;
    // rule 1: wander (toroidal). choice((-1,0,1)) consumes the RNG per axis.
    this.x = (this.x + n.rng.choice([-1, 0, 1]) + n.w) % n.w;
    this.y = (this.y + n.rng.choice([-1, 0, 1]) + n.h) % n.h;
    const here = n.grid[this.y][this.x];
    this._push(here);                                 // rule 2 (record even empties)
    if (this.carry === null) {
      if (here !== null) {                            // rule 3: maybe pick up
        const f = this._f(here);
        const p = Math.pow(this.kp / (this.kp + f), 2); // PAPER §3.2 VERBATIM: (k+/(k++f))^2
        if (n.rng.random() < p) {
          this.carry = here; n.grid[this.y][this.x] = null;
        }
      }
    } else {
      if (here === null) {                            // rule 4: maybe drop
        const f = this._f(this.carry);
        const p = Math.pow(f / (this.km + f), 2);     // PAPER §3.2 VERBATIM: (f/(k-+f))^2
        if (n.rng.random() < p) {
          n.grid[this.y][this.x] = this.carry; this.carry = null;
        }
      }
    }
  }
}

function render(nest) {
  const out = [];
  for (const row of nest.grid) out.push(row.map((c) => (c ? c : '.')).join(''));
  console.log(out.join('\n'));
}

function run(ticks = 120000, nAnts = 40, seed = 0, verbose = true) {
  const nest = new Nest(40, 24, 90, seed);
  const ants = Array.from({ length: nAnts }, () => new SortAnt(nest));
  if (verbose) {
    console.log('BEFORE (random scatter):\n');
    render(nest);
    console.log(`\ninitial clustering: ${nest.clustering().toFixed(3)}`);
  }
  const hist = [];
  const stride = Math.max(1, Math.floor(ticks / 12));
  for (let t = 0; t < ticks; t++) {
    for (const a of ants) a.step();
    if (t % stride === 0) hist.push(nest.clustering());
  }
  if (verbose) {
    console.log('\nAFTER (emergent sorting):\n');
    render(nest);
    console.log(`\nfinal clustering: ${nest.clustering().toFixed(3)}`);
    console.log('clustering(t): ' + hist.map((c) => c.toFixed(2)).join(' '));
  }
  return { nest, hist };
}

function main() {
  const args = process.argv.slice(2);
  const opts = { ticks: 120000, ants: 40, seed: 0 };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--ticks') opts.ticks = parseInt(args[++i], 10);
    else if (a === '--ants') opts.ants = parseInt(args[++i], 10);
    else if (a === '--seed') opts.seed = parseInt(args[++i], 10);
  }
  run(opts.ticks, opts.ants, opts.seed);
}

main();
