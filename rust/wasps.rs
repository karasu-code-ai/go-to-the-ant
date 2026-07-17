// 'Go to the Ant' §3.4 — Wasp Task Differentiation (Theraulaz et al. 1991), a faithful
// Rust port from the paper.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
// Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
//
// Mature Polistes wasps — genetically IDENTICAL — split into a single Chief, a band of
// Foragers, and a band of Nurses, with no HR department and no wasp computing the
// proportion. Parunak's three interacting rules:
//   1. FACE-OFFS. When two wasps meet, j beats i with the Fermi probability
//      p = 1/(1 + e^(h·(F_i − F_j))). The higher force usually wins (but not always);
//      a quantum of Force passes loser → winner.
//   2. BROOD DEMAND.  D(t) = D(t−1) + appetite − W, where W is the food-work by foragers.
//   3. FORAGE?  A wasp near the brood forages with Fermi prob  p = 1/(1 + e^(hf·(σ_j − D))).
//      Foraging LOWERS its threshold σ by ξ (learning); not foraging RAISES σ by φ (forget).
//
// Force is MOBILITY (a low-force wasp is stimulated by the brood but cannot travel to hunt).
// The joint (Force, Threshold) distribution self-separates into three castes:
//   · Foragers  = high force, low threshold  (strong enough to move + sensitive to the brood)
//   · Nurses    = low force,  low threshold  (attentive, but stuck near the brood)
//   · Chief     = one wasp, high force, high threshold (grounds the scales; doesn't forage)
//
// The RUST LENS: ownership and borrowing make "who is allowed to mutate the shared state
// this step" explicit and machine-checked. The colony's force/threshold vectors live in one
// `Colony` struct; each rule borrows `&mut self` for the whole step, so the three coupled
// rules run in a strictly serial, single-owner order — the borrow checker guarantees no rule
// aliases the shared force vector while another is mutating it. That serial discipline is
// exactly the Python semantics, made explicit rather than assumed.
//
// std-only, single file. Build:  rustc -O wasps.rs   Run: ./wasps --seed 0

// --- PRNG: SplitMix64, identical across all ports so results are directly comparable. ---
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    // [0,1): (next() >> 11) * 2^-53
    fn random_float(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
    // randrange(n) -> [0,n)
    fn randrange(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    // uniform(a,b): a + random_float()*(b-a)
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + self.random_float() * (b - a)
    }
}

struct Colony {
    n: usize,
    f: Vec<f64>,   // Force = mobility
    sig: Vec<f64>, // Threshold σ
    seenmax: Vec<f64>, // per-wasp FADING memory of the top force it has faced (LOCAL, no global max)
    d: f64,        // brood demand D
    h: f64,
    hf: f64,
    q: f64, // quantum
    appetite: f64,
    xi: f64,
    phi: f64,
    mob: f64,
    leak: f64,
    gen: f64,
    seendecay: f64, // the seenmax memory fades (robustness)
    rng: SplitMix64,
}

impl Colony {
    const SIGMAX: f64 = 4.0;
    // NOTE (provenance, honest labelling): the genuine §4.6 entropy leak for wasps is RULE 1's
    // conservative force TRANSFER (Parunak names "the flow of force among wasps"). The leak/gen
    // term below is a SEPARATE Force-RELAXATION (mean-reversion to ~gen/leak) — an INFERENCE
    // BEYOND Parunak (plausibly a Theraulaz 1991 element we can't verify), replacing an ad-hoc
    // force CAP. It is empirically required: pure conservation condenses to one super-wasp with
    // no graded forager band. The dominance term is now LOCAL (per-wasp seenmax). See PROVENANCE.md.
    fn new(n: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        // genetically identical: tiny initial spread only.
        // NOTE: consume the RNG in EXACTLY the Python order — all F's first, then all sig's.
        let mut f = Vec::with_capacity(n);
        for _ in 0..n {
            f.push(1.0 + rng.uniform(-0.05, 0.05));
        }
        let mut sig = Vec::with_capacity(n);
        for _ in 0..n {
            sig.push(1.6 + rng.uniform(-0.05, 0.05));
        }
        let seenmax = f.clone();
        Colony {
            n,
            f,
            sig,
            seenmax,
            d: 2.0,
            h: 1.1,
            hf: 3.0,
            q: 0.10,
            // appetite sized so the mobile minority (~n/8) can ALMOST meet demand — leaving the
            // top wasp (the Chief) surplus, so its threshold drifts high while foragers stay low.
            appetite: 0.075 * n as f64,
            xi: 0.02,
            phi: 0.012,
            mob: 1.6,
            leak: 0.004,
            gen: 0.005,
            seendecay: 0.998,
            rng,
        }
    }

    fn step(&mut self) {
        let n = self.n;
        // rule 1: face-offs — gentle, capped, so a graded hierarchy forms (not one super-wasp)
        for _ in 0..(n / 3) {
            let i = self.rng.randrange(n);
            let j = self.rng.randrange(n);
            if i == j {
                continue;
            }
            // PAPER §3.4 VERBATIM: p = 1/(1 + e^(h(Fi-Fj)))
            let (fi, fj) = (self.f[i], self.f[j]);
            let pj = 1.0 / (1.0 + (self.h * (fi - fj)).exp());
            let (w, l) = if self.rng.random_float() < pj { (j, i) } else { (i, j) };
            let t = self.q.min(self.f[l]);
            self.f[w] += t;
            self.f[l] -= t; // force is conserved in the face-off (paper)
            let m = if fi > fj { fi } else { fj }; // LOCAL: each wasp's FADING memory of the strongest force faced
            let di = self.seenmax[i] * self.seendecay;
            let dj = self.seenmax[j] * self.seendecay;
            self.seenmax[i] = if m > di { m } else { di };
            self.seenmax[j] = if m > dj { m } else { dj };
        }
        // FORCE RELAXATION (inference beyond Parunak, NOT the §4.6 entropy leak — that is Rule 1's
        // conservative force flow above): force mean-reverts toward ~gen/leak each tick. A steady
        // leak+gen bounds the hierarchy naturally, so no ad-hoc force cap is needed to stop one
        // super-wasp. Empirically required (pure conservation condenses to one super-wasp).
        for k in 0..n {
            self.f[k] = (self.f[k] * (1.0 - self.leak) + self.gen).max(0.0);
        }
        // rules 2 & 3: brood stimulation + foraging. Work = COUNT of mobile foragers (each 1).
        // SPATIALITY PROXY (operationalized, now LOCAL): the paper's Chief "wanders and faces off",
        // so it is NOT near the brood and is rarely stimulated -> its threshold drifts HIGH. We
        // approximate "away dominating" by suppressing foraging in proportion to (F/seenmax)^4,
        // where seenmax is each wasp's OWN fading memory of the top force it has faced (NO global
        // max) -> ~1 only for the wasp atop its own encounters (the Chief). Restores its high-σ.
        let mut w_count: u32 = 0;
        for k in 0..n {
            // PAPER §3.4 VERBATIM: p = 1/(1 + e^(hf(sig-D)))
            let pf = 1.0 / (1.0 + (self.hf * (self.sig[k] - self.d)).exp());
            let sm = if self.seenmax[k] > 0.0 { self.seenmax[k] } else { 1.0 };
            let ratio = self.f[k] / sm;
            let dom = ratio * ratio * ratio * ratio; // (F/seenmax)^4 ~1 only for the wasp atop its own encounters
            if self.rng.random_float() < pf * (1.0 - dom) {
                // stimulated AND not away dominating
                self.sig[k] = (self.sig[k] - self.xi).max(0.0); // learns: threshold drops
                if self.f[k] > self.mob {
                    w_count += 1; // mobile enough to actually hunt
                }
            } else {
                self.sig[k] = (self.sig[k] + self.phi).min(Self::SIGMAX); // forgets: threshold rises
            }
        }
        self.d = (self.d + self.appetite - w_count as f64).max(0.0);
    }

    // Returns (chief_index, caste_of_each_wasp) where caste 0=Chief, 1=Forager, 2=Nurse.
    fn castes(&self) -> (usize, Vec<u8>) {
        let mut chief = 0usize;
        for k in 1..self.n {
            if self.f[k] > self.f[chief] {
                chief = k;
            }
        }
        // median threshold (Python: sorted(sig)[n//2])
        let mut ssort = self.sig.clone();
        ssort.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let smed = ssort[self.n / 2];
        let mut castes = vec![2u8; self.n];
        for k in 0..self.n {
            if k == chief {
                castes[k] = 0;
            } else if self.f[k] > self.mob && self.sig[k] <= smed {
                castes[k] = 1; // mobile + responsive
            } else {
                castes[k] = 2; // immobile (or unresponsive) -> stays with the brood
            }
        }
        (chief, castes)
    }
}

fn run(ticks: usize, n: usize, seed: u64) {
    let mut c = Colony::new(n, seed);
    let mut hist: Vec<(usize, usize)> = Vec::new();
    let sample_every = std::cmp::max(1, ticks / 12);
    for t in 0..ticks {
        c.step();
        if t % sample_every == 0 {
            let (_, castes) = c.castes();
            let f = castes.iter().filter(|&&x| x == 1).count();
            let ns = castes.iter().filter(|&&x| x == 2).count();
            hist.push((f, ns));
        }
    }

    let (chief, castes) = c.castes();
    println!(
        "Emergent castes from {} genetically identical wasps ({} ticks):\n",
        n, ticks
    );
    let names = ["Chief", "Forager", "Nurse"];
    for (code, name) in names.iter().enumerate() {
        let ks: Vec<usize> = (0..n).filter(|&k| castes[k] as usize == code).collect();
        if ks.is_empty() {
            continue;
        }
        let mf: f64 = ks.iter().map(|&k| c.f[k]).sum::<f64>() / ks.len() as f64;
        let ms: f64 = ks.iter().map(|&k| c.sig[k]).sum::<f64>() / ks.len() as f64;
        println!(
            "  {:8} n={:3}   mean Force {:5.2}   mean Threshold {:5.2}",
            name,
            ks.len(),
            mf,
            ms
        );
    }
    let popmean: f64 = c.f.iter().sum::<f64>() / n as f64;
    println!(
        "\n  Chief force {:.2} (pop mean {:.2}), threshold {:.2}",
        c.f[chief], popmean, c.sig[chief]
    );
    print!("  Forager/Nurse split(t):");
    for (f, ns) in hist.iter() {
        print!(" {}/{}", f, ns);
    }
    println!();
    landscape(&c);
}

// ASCII scatter of the population in (Force -> x, Threshold -> y) space.
fn landscape(c: &Colony) {
    let cols = 48usize;
    let rows = 16usize;
    let fmn = c.f.iter().cloned().fold(f64::MAX, f64::min);
    let fmx = c.f.iter().cloned().fold(f64::MIN, f64::max);
    let smn = c.sig.iter().cloned().fold(f64::MAX, f64::min);
    let smx = c.sig.iter().cloned().fold(f64::MIN, f64::max);
    let mut grid = vec![vec![' '; cols]; rows];
    let mut chief = 0usize;
    for k in 1..c.n {
        if c.f[k] > c.f[chief] {
            chief = k;
        }
    }
    let mut fsort = c.f.clone();
    fsort.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fmed = fsort[c.n / 2];
    for k in 0..c.n {
        let x = ((c.f[k] - fmn) / (fmx - fmn + 1e-9) * (cols - 1) as f64) as usize;
        let y = ((c.sig[k] - smn) / (smx - smn + 1e-9) * (rows - 1) as f64) as usize;
        let mark = if k == chief {
            'C'
        } else if c.f[k] >= fmed {
            'F'
        } else {
            'n'
        };
        grid[rows - 1 - y][x] = mark;
    }
    println!("\n  (F,\u{3c3}) landscape — x = Force \u{2192}, y = Threshold \u{2191} | C chief, F forager, n nurse:\n");
    for row in grid.iter() {
        let s: String = row.iter().collect();
        println!("   {}", s);
    }
}

fn main() {
    let mut ticks: usize = 4000;
    let mut wasps: usize = 80;
    let mut seed: u64 = 0;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--ticks" => {
                i += 1;
                ticks = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(4000);
            }
            // Python calls this dimension --wasps; accept --ants too for cross-port CLI parity.
            "--wasps" | "--ants" => {
                i += 1;
                wasps = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(80);
            }
            _ => {}
        }
        i += 1;
    }

    run(ticks, wasps, seed);
}
