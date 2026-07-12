#!/usr/bin/env node
/*
 * "Go to the Ant" — a faithful port of Parunak's OG foraging swarm (§3.1).
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
 *
 * The five local ant rules (VERBATIM in spirit):
 *   1. Avoid obstacles.
 *   2. Wander randomly, biased toward nearby pheromone (Brownian floor keeps the
 *      walk alive even on a trail — supporting entropy cuts the short-cuts).
 *   3. If carrying food, drop pheromone at a CONSTANT RATE while walking.
 *   4. If at food and empty-handed, pick it up.
 *   5. If at the nest and carrying, drop it.
 *
 * TWO local pheromone fields — the whole point is stigmergy, communication
 * THROUGH THE ENVIRONMENT (the fields are the only channel; no ant knows where
 * the nest is — carriers just climb the local home gradient):
 *   food_pher: laid by CARRIERS, followed by SEARCHERS.
 *   home_pher: emitted + diffused by the NEST, followed by CARRIERS.
 * Evaporation every tick is the entropy leak: trails laid by ants who never got
 * home — and trails to depleted sources — fade. No ant plans a route; the
 * network EMERGES from deposit + diffusion + evaporation + weighted-random walk.
 *
 * LENS (JavaScript): zero-dependency, standard-library only. Runs under node and
 * is portable to the browser visualizer — the web-native, accessible lens. All
 * 64-bit PRNG arithmetic uses BigInt (JS numbers are IEEE doubles); the sim math
 * uses ordinary doubles, exactly as the Python reference does.
 */

'use strict';

// The 8 neighbours, in THIS exact order (matters for the weighted-random pick).
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
}

class World {
  constructor(seed = 0, useWall = false, w = 56, h = 28, region = 2) {
    this.w = w; this.h = h; this.region = region;
    this.rng = new SplitMix64(seed);
    // TWO local fields — not a global homing beacon.
    this.food_pher = Array.from({ length: h }, () => new Float64Array(w));
    this.home_pher = Array.from({ length: h }, () => new Float64Array(w));
    this.obstacle = Array.from({ length: h }, () => new Array(w).fill(false));
    this.nest = [5, h >> 1];          // (5, 14)
    this.food = [w - 6, h >> 1];      // (50, 14)
    this.food_qty = 1e9;              // effectively unlimited source
    this.gap = null;
    if (useWall) {                    // optional wall with a single gap -> must ROUTE
      const wallx = w >> 1;
      // rng.randint(4, h-5) inclusive -> uniform over that inclusive range
      const lo = 4, hi = h - 5;
      this.gap = lo + Number(this.rng.next() % BigInt(hi - lo + 1));
      for (let y = 0; y < h; y++) {
        if (Math.abs(y - this.gap) > 2) this.obstacle[y][wallx] = true;
      }
    }
    this.deliveries = 0;
  }

  free(x, y) {
    return x >= 0 && x < this.w && y >= 0 && y < this.h && !this.obstacle[y][x];
  }
  at_food(x, y) {
    return Math.abs(x - this.food[0]) <= this.region && Math.abs(y - this.food[1]) <= this.region;
  }
  at_nest(x, y) {
    return Math.abs(x - this.nest[0]) <= this.region && Math.abs(y - this.nest[1]) <= this.region;
  }

  evaporate(rate) {                   // both fields dissipate (the entropy leak)
    const keep = 1.0 - rate;
    for (let y = 0; y < this.h; y++) {
      const fr = this.food_pher[y], hr = this.home_pher[y];
      for (let x = 0; x < this.w; x++) { fr[x] *= keep; hr[x] *= keep; }
    }
  }

  // The NEST is a home-pheromone SOURCE; the marker DIFFUSES outward (Brownian)
  // into a gradient that points home from everywhere. One Jacobi pass (simultaneous
  // update via a copy), matching the Python.
  emit_and_diffuse_home() {
    const [nx, ny] = this.nest;
    for (let dy = -this.region; dy <= this.region; dy++) {
      for (let dx = -this.region; dx <= this.region; dx++) {
        const x = nx + dx, y = ny + dy;
        if (this.free(x, y)) this.home_pher[y][x] += 6.0;
      }
    }
    const nxt = this.home_pher.map((row) => Float64Array.from(row));
    for (let y = 0; y < this.h; y++) {
      for (let x = 0; x < this.w; x++) {
        if (!this.free(x, y)) continue;
        let s = this.home_pher[y][x], c = 1;
        for (const [dx, dy] of DIRS) {
          const xx = x + dx, yy = y + dy;
          if (this.free(xx, yy)) { s += this.home_pher[yy][xx]; c += 1; }
        }
        nxt[y][x] = s / c;
      }
    }
    this.home_pher = nxt;
  }
}

class Ant {
  constructor(world) {
    this.w = world;
    this.x = world.nest[0]; this.y = world.nest[1];
    this.carrying = false;
  }

  // Rule 2, fully LOCAL: follow the field that leads where you're going.
  // Searchers read FOOD scent (toward food); carriers read HOME scent (toward nest).
  _weights() {
    const field = this.carrying ? this.w.home_pher : this.w.food_pher;
    const wts = new Array(8);
    for (let i = 0; i < 8; i++) {
      const nx = this.x + DIRS[i][0], ny = this.y + DIRS[i][1];
      if (!this.w.free(nx, ny)) { wts[i] = 0.0; continue; }   // rule 1: never step into a wall
      wts[i] = 1.0 + field[ny][nx] * 6.0;                     // Brownian floor + local scent bias
    }
    return wts;
  }

  step(deposit) {
    const wts = this._weights();
    let tot = 0.0;
    for (let i = 0; i < 8; i++) tot += wts[i];
    if (tot <= 0) return;                                     // boxed in — stay put this tick
    const r = this.w.rng.random() * tot;
    let acc = 0.0;
    for (let i = 0; i < 8; i++) {
      acc += wts[i];
      if (r <= acc) { this.x += DIRS[i][0]; this.y += DIRS[i][1]; break; }  // note: <=, matches Python
    }
    // rule 3: carriers lay the FOOD trail (searchers lay nothing; the nest broadcasts HOME)
    if (this.carrying) this.w.food_pher[this.y][this.x] += deposit;
    // rule 4: pick up food
    if (this.w.at_food(this.x, this.y) && !this.carrying && this.w.food_qty > 0) {
      this.carrying = true; this.w.food_qty -= 1;
    // rule 5: drop food at the nest
    } else if (this.w.at_nest(this.x, this.y) && this.carrying) {
      this.carrying = false; this.w.deliveries += 1;
    }
  }
}

function run(ticks, nAnts, evap, deposit, seed, useWall) {
  const world = new World(seed, useWall);
  const ants = Array.from({ length: nAnts }, () => new Ant(world));
  const history = [];
  const stride = Math.max(1, Math.floor(ticks / 20));
  for (let t = 0; t < ticks; t++) {
    world.emit_and_diffuse_home();                           // nest broadcasts the home gradient
    for (const a of ants) a.step(deposit);
    world.evaporate(evap);
    if (t % stride === 0) history.push([t, world.deliveries]);
  }
  return { world, ants, history };
}

function renderAscii(world, ants) {
  let peak = 0.0;
  for (let y = 0; y < world.h; y++) for (let x = 0; x < world.w; x++)
    if (world.food_pher[y][x] > peak) peak = world.food_pher[y][x];
  if (peak === 0.0) peak = 1.0;
  const shades = " .:-=+*#%@";
  const antpos = new Set(ants.map((a) => a.x + ',' + a.y));
  const out = [];
  out.push("\nGo to the Ant — emergent trail (nest N <-> food F, wall '|', pheromone density):\n");
  for (let y = 0; y < world.h; y++) {
    let line = '';
    for (let x = 0; x < world.w; x++) {
      let c;
      if (x === world.nest[0] && y === world.nest[1]) c = 'N';
      else if (x === world.food[0] && y === world.food[1]) c = 'F';
      else if (world.obstacle[y][x]) c = '|';
      else if (antpos.has(x + ',' + y)) c = 'o';
      else {
        let lvl = Math.floor((world.food_pher[y][x] / peak) * (shades.length - 1));
        lvl = Math.max(0, Math.min(shades.length - 1, lvl));
        c = shades[lvl];
      }
      line += c;
    }
    out.push(line);
  }
  console.log(out.join('\n'));
}

function main() {
  const args = process.argv.slice(2);
  const opts = { ticks: 3000, ants: 90, evap: 0.015, deposit: 1.0, seed: 0, wall: false };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--wall') opts.wall = true;
    else if (a === '--ticks') opts.ticks = parseInt(args[++i], 10);
    else if (a === '--ants') opts.ants = parseInt(args[++i], 10);
    else if (a === '--evap') opts.evap = parseFloat(args[++i]);
    else if (a === '--deposit') opts.deposit = parseFloat(args[++i]);
    else if (a === '--seed') opts.seed = parseInt(args[++i], 10);
  }
  const { world, ants, history } = run(opts.ticks, opts.ants, opts.evap, opts.deposit, opts.seed, opts.wall);
  renderAscii(world, ants);
  console.log(`\nfood delivered to nest over ${opts.ticks} ticks: ${world.deliveries}`);
  console.log('deliveries(t): ' + history.map(([, d]) => d).join(' '));
}

main();
