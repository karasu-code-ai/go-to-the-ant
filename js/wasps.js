#!/usr/bin/env node
/*
 * "Go to the Ant" §3.4 — Wasp Task Differentiation (Theraulaz et al. 1991),
 * recreated from the paper.
 *
 * Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from
 * Natural Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
 *
 * Mature Polistes wasps — genetically IDENTICAL — split into a single Chief, a
 * band of Foragers, and a band of Nurses, with no HR department and no wasp
 * computing the proportion. Parunak's three interacting rules:
 *
 *   1. FACE-OFFS. When two wasps meet, j beats i with the Fermi probability
 *      p = 1/(1 + e^(h*(F_i - F_j))). The higher force usually wins (but not
 *      always); a quantum of Force passes loser -> winner.
 *   2. BROOD DEMAND. D(t) = D(t-1) + appetite - W, where W is the food-work
 *      done by all foragers.
 *   3. FORAGE? A wasp near the brood forages with Fermi probability
 *      p = 1/(1 + e^(hf*(sig_j - D))). Foraging LOWERS its threshold sig by xi
 *      (learning); not foraging RAISES sig by phi (forgetting).
 *
 * Force is MOBILITY (a low-force wasp is stimulated by the brood but cannot
 * travel to hunt). The joint (Force, Threshold) distribution self-separates
 * into three clusters — the castes:
 *   . Foragers  = high force, low threshold  (strong enough to move + sensitive)
 *   . Nurses    = low force,  low threshold  (attentive, but stuck near brood)
 *   . Chief     = one wasp, high force, high threshold (grounds the scales)
 *
 * LENS (JavaScript): zero-dependency, standard-library only. Runs under node and
 * is portable straight into the browser visualizer — the web-native lens. All
 * 64-bit PRNG arithmetic uses BigInt (JS numbers are IEEE doubles); the field
 * math uses ordinary doubles, exactly as the Python reference does.
 */

'use strict';

// ---- PRNG: SplitMix64, identical across all ports (BigInt for u64 wrapping) ----
// CROSS-PORT CONVENTION: this intentionally does NOT reproduce CPython's random
// module; all six ports agree with EACH OTHER via this shared PRNG.
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
  // uniform(a, b) -> double in [a, b)
  uniform(a, b) { return a + this.random() * (b - a); }
}

class Colony {
  // NOTE (provenance, honest labelling): the genuine §4.6 entropy leak for wasps is
  // RULE 1's conservative force TRANSFER (Parunak names "the flow of force among wasps").
  // The leak/gen term below is a SEPARATE Force-RELAXATION (mean-reversion to ~gen/leak)
  // — an INFERENCE BEYOND Parunak (plausibly a Theraulaz 1991 element we can't verify),
  // replacing an ad-hoc force CAP. Empirically required: pure conservation condenses to
  // one super-wasp with no graded forager band. The dominance term is now LOCAL (seenmax).
  constructor(n = 80, seed = 0, h = 1.1, hf = 3.0, quantum = 0.10, appetite = null,
              xi = 0.02, phi = 0.012, mob = 1.6, leak = 0.004, gen = 0.005) {
    this.SIGMAX = 4.0;
    this.rng = new SplitMix64(seed);
    this.n = n;
    // genetically identical: tiny initial spread only.
    // NOTE: consume RNG in exactly the Python order — all n F-uniforms first,
    // then all n sig-uniforms — so the sequence stays cross-port identical.
    this.F = new Array(n);
    for (let i = 0; i < n; i++) this.F[i] = 1.0 + this.rng.uniform(-0.05, 0.05);
    this.sig = new Array(n);
    for (let i = 0; i < n; i++) this.sig[i] = 1.6 + this.rng.uniform(-0.05, 0.05);
    this.seenmax = this.F.slice();   // per-wasp FADING memory of the top force it has faced (LOCAL)
    this.seendecay = 0.998;          // the memory fades (robustness)
    this.D = 2.0;
    this.h = h; this.hf = hf; this.q = quantum;
    // appetite sized so the mobile minority (~n/8) can ALMOST meet demand —
    // leaving the top wasp (the Chief) surplus, so its threshold drifts high
    // while the working foragers stay low.
    this.appetite = appetite !== null ? appetite : 0.075 * n;
    this.xi = xi; this.phi = phi; this.mob = mob;
    this.leak = leak; this.gen = gen;
  }

  step() {
    const rng = this.rng, F = this.F, sig = this.sig, n = this.n;
    // rule 1: face-offs — gentle, capped, so a graded hierarchy forms.
    for (let f = 0; f < Math.floor(n / 3); f++) {
      const i = rng.randrange(n), j = rng.randrange(n);
      if (i === j) continue;
      // PAPER §3.4 VERBATIM: p = 1/(1 + e^(h*(Fi - Fj)))
      const fi = F[i], fj = F[j];
      const pj = 1.0 / (1.0 + Math.exp(this.h * (fi - fj)));
      let w, l;
      if (rng.random() < pj) { w = j; l = i; } else { w = i; l = j; }
      const t = Math.min(this.q, F[l]);
      F[w] += t; F[l] -= t;                    // force is conserved in the face-off (paper)
      const m = fi > fj ? fi : fj;             // LOCAL: each wasp's FADING memory of the strongest force faced
      const di = this.seenmax[i] * this.seendecay, dj = this.seenmax[j] * this.seendecay;
      this.seenmax[i] = m > di ? m : di;
      this.seenmax[j] = m > dj ? m : dj;
    }
    // FORCE RELAXATION (inference beyond Parunak, NOT the §4.6 entropy leak — that is
    // Rule 1's conservative force flow above): force mean-reverts toward ~gen/leak each
    // tick. A steady leak+gen bounds the hierarchy naturally, so no ad-hoc force cap is
    // needed. Empirically required (pure conservation condenses to one super-wasp).
    for (let k = 0; k < n; k++) {
      F[k] = Math.max(0.0, F[k] * (1.0 - this.leak) + this.gen);
    }
    // rules 2 & 3: brood stimulation + foraging. Work = COUNT of mobile foragers.
    // SPATIALITY PROXY (OPERATIONALIZED, now LOCAL): the paper's Chief "wanders and
    // faces off", so it is NOT near the brood and is rarely stimulated -> its threshold
    // drifts HIGH. We approximate "away dominating" via dominance = (F/seenmax)^4, where
    // seenmax is each wasp's OWN fading memory of the top force it has faced (NO global
    // max) -> ~1 only for the wasp atop its own encounters (the Chief). Restores its high-sig.
    let W = 0;
    for (let k = 0; k < n; k++) {
      // PAPER §3.4 VERBATIM: p = 1/(1 + e^(hf*(sig - D)))
      const pf = 1.0 / (1.0 + Math.exp(this.hf * (sig[k] - this.D)));
      const sm = this.seenmax[k] > 0.0 ? this.seenmax[k] : 1.0;
      const ratio = F[k] / sm;
      const dom = ratio * ratio * ratio * ratio;        // (F/seenmax)^4 ~1 only for the wasp atop its own encounters
      if (rng.random() < pf * (1.0 - dom)) {            // stimulated AND not away dominating
        sig[k] = Math.max(0.0, sig[k] - this.xi);       // learns: threshold drops
        if (F[k] > this.mob) W += 1;                    // mobile enough to actually hunt
      } else {
        sig[k] = Math.min(this.SIGMAX, sig[k] + this.phi);  // forgets: threshold rises
      }
    }
    this.D = Math.max(0.0, this.D + this.appetite - W);
  }

  castes() {
    const F = this.F, sig = this.sig, n = this.n;
    let chief = 0;
    for (let k = 1; k < n; k++) if (F[k] > F[chief]) chief = k;
    const smed = [...sig].sort((a, b) => a - b)[Math.floor(n / 2)];
    const groups = { Chief: [], Forager: [], Nurse: [] };
    for (let k = 0; k < n; k++) {
      if (k === chief) groups.Chief.push(k);
      else if (F[k] > this.mob && sig[k] <= smed) groups.Forager.push(k);  // mobile + responsive
      else groups.Nurse.push(k);                                          // immobile/unresponsive
    }
    return { groups, chief };
  }
}

function run(ticks, n, seed, verbose = true) {
  const c = new Colony(n, seed);
  const hist = [];
  const stride = Math.max(1, Math.floor(ticks / 12));
  for (let t = 0; t < ticks; t++) {
    c.step();
    if (t % stride === 0) {
      const { groups } = c.castes();
      hist.push([groups.Forager.length, groups.Nurse.length]);
    }
  }
  if (verbose) {
    const { groups, chief } = c.castes();
    console.log(`Emergent castes from ${n} genetically identical wasps (${ticks} ticks):\n`);
    for (const name of ['Chief', 'Forager', 'Nurse']) {
      const ks = groups[name];
      if (ks.length === 0) continue;
      let mF = 0, mS = 0;
      for (const k of ks) { mF += c.F[k]; mS += c.sig[k]; }
      mF /= ks.length; mS /= ks.length;
      console.log(`  ${name.padEnd(8)} n=${String(ks.length).padStart(3)}   ` +
        `mean Force ${mF.toFixed(2).padStart(5)}   mean Threshold ${mS.toFixed(2).padStart(5)}`);
    }
    const popMean = c.F.reduce((a, b) => a + b, 0) / n;
    console.log(`\n  Chief force ${c.F[chief].toFixed(2)} (pop mean ${popMean.toFixed(2)}), ` +
      `threshold ${c.sig[chief].toFixed(2)}`);
    console.log('  Forager/Nurse split(t): ' + hist.map(([f, ns]) => `${f}/${ns}`).join(' '));
    landscape(c);
  }
  return c;
}

function landscape(c, cols = 48, rows = 16) {
  // ASCII scatter of the population in (Force -> x, Threshold -> y) space.
  const F = c.F, sig = c.sig, n = c.n;
  let fmn = Infinity, fmx = -Infinity, smn = Infinity, smx = -Infinity;
  for (let k = 0; k < n; k++) {
    if (F[k] < fmn) fmn = F[k]; if (F[k] > fmx) fmx = F[k];
    if (sig[k] < smn) smn = sig[k]; if (sig[k] > smx) smx = sig[k];
  }
  const grid = Array.from({ length: rows }, () => new Array(cols).fill(' '));
  let chief = 0;
  for (let k = 1; k < n; k++) if (F[k] > F[chief]) chief = k;
  const fmed = [...F].sort((a, b) => a - b)[Math.floor(n / 2)];
  for (let k = 0; k < n; k++) {
    const x = Math.floor((F[k] - fmn) / (fmx - fmn + 1e-9) * (cols - 1));
    const y = Math.floor((sig[k] - smn) / (smx - smn + 1e-9) * (rows - 1));
    const mark = k === chief ? 'C' : (F[k] >= fmed ? 'F' : 'n');
    grid[rows - 1 - y][x] = mark;
  }
  console.log('\n  (F,sig) landscape — x = Force ->, y = Threshold ^ | C chief, F forager, n nurse:\n');
  for (const row of grid) console.log('   ' + row.join(''));
}

function main() {
  const args = process.argv.slice(2);
  const opts = { ticks: 4000, wasps: 80, seed: 0 };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--ticks') opts.ticks = parseInt(args[++i], 10);
    else if (a === '--wasps' || a === '--ants') opts.wasps = parseInt(args[++i], 10);
    else if (a === '--seed') opts.seed = parseInt(args[++i], 10);
  }
  run(opts.ticks, opts.wasps, opts.seed);
}

main();
