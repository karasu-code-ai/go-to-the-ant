// 'Go to the Ant' §3.6 — Wolves: Surrounding Prey (Korf 1992), a faithful Rust port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
// Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997), §3.6, after
// R. Korf, "A simple solution to pursuit games" (1992).
//
// One wolf can't kill a moose; the pack must SURROUND it — with no radios and no
// negotiated strategy ("I'll take the north side, you take the south"). Parunak gives
// two local rules (§3.6):
//   1. MOOSE: move to the neighbouring cell FARTHEST from the nearest wolf. (If it were
//      faster than the wolves it escapes; here it isn't.)
//   2. WOLVES: move to minimise  S = d(moose) - k*d(nearest other wolf)  — get CLOSE to
//      the moose while staying FAR from each other. k is a repulsion tuning constant.
// With attraction (to prey) and repulsion (between wolves) balanced, the pack inevitably
// encircles the moose, no communication required.
//
// PROVENANCE:
//   VERBATIM (paper, Korf 1992): the two rules and the score  S = d(moose) - k*d(wolf).
//   OPERATIONALIZED: speeds (vm=0.6, vw=1.0), k=1.12, and a continuous plane with a
//     24-candidate-direction search instead of the paper's hex grid. The paper states
//     six wolves capture on a hex grid; here a continuous plane, candidate-direction
//     search. There is NO per-step randomness — the hunt is deterministic given the
//     random initial wolf placement.
//
// The RUST LENS: ownership and borrowing make "who is allowed to mutate the shared state
// this step" explicit and machine-checked. Each step is two phases that mutate the shared
// `Hunt` state: (1) the moose reads all wolves and moves; (2) each wolf reads the moose
// and all OTHER wolves and moves. The Python mutates the moose in place, then reads the
// OLD wolf positions while building a fresh `nw` vector and swaps it in at the end. Here
// that read/write split is enforced: the wolf phase borrows the old `wolves` slice
// immutably to compute, writing into a separate `next` Vec that replaces it only after the
// whole phase completes — the borrow checker guarantees no wolf sees a half-updated pack,
// making the Python's implicit simultaneous-wolf-update explicit and alias-free.
//
// std-only, single file. Build:  rustc -O wolves.rs    Run: ./wolves --seed 0

use std::f64::consts::PI;

const TAU: f64 = 2.0 * PI;

// --- PRNG: SplitMix64, identical across all ports so results are directly comparable. ---
// This is a CROSS-PORT CONVENTION; it does NOT reproduce CPython's random module (the
// reference uses Mersenne-Twister). The goal is that the six sequential ports agree with
// EACH OTHER given the same --seed, not with the Python numbers.
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
    // uniform(a,b): a + random_float()*(b-a)
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + self.random_float() * (b - a)
    }
}

struct Hunt {
    w: f64,
    h: f64,
    vm: f64,
    vw: f64,
    k: f64,
    mx: f64,
    my: f64,
    wolves: Vec<(f64, f64)>,
}

impl Hunt {
    fn new(n_wolves: usize, w: f64, h: f64, seed: u64, vm: f64, vw: f64, k: f64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mx = w / 2.0;
        let my = h / 2.0;
        // Consume the RNG in EXACTLY the Python order: per wolf, uniform(0,w) then
        // uniform(0,h). (Python: [(rng.uniform(0,w), rng.uniform(0,h)) for _ in range(n)].)
        let mut wolves = Vec::with_capacity(n_wolves);
        for _ in 0..n_wolves {
            let wx = rng.uniform(0.0, w);
            let wy = rng.uniform(0.0, h);
            wolves.push((wx, wy));
        }
        Hunt { w, h, vm, vw, k, mx, my, wolves }
    }

    // The 24 directional candidates PLUS staying put, in the Python index order
    // (a = i*TAU/24 for i in 0..23, then (x,y)).
    fn cands(x: f64, y: f64, sp: f64) -> Vec<(f64, f64)> {
        let mut v = Vec::with_capacity(25);
        for i in 0..24 {
            let a = i as f64 * TAU / 24.0;
            v.push((x + sp * a.cos(), y + sp * a.sin()));
        }
        v.push((x, y));
        v
    }

    fn in_bounds(&self, cx: f64, cy: f64) -> bool {
        cx >= 0.0 && cx < self.w && cy >= 0.0 && cy < self.h
    }

    fn step(&mut self) -> f64 {
        // Rule 1 (VERBATIM): the moose flees to the in-bounds candidate whose nearest-wolf
        // distance is MAXIMAL.
        let mut best = (self.mx, self.my);
        let mut bd = -1.0f64;
        for (cx, cy) in Hunt::cands(self.mx, self.my, self.vm) {
            if !self.in_bounds(cx, cy) {
                continue;
            }
            let mut d = f64::INFINITY;
            for &(wx, wy) in self.wolves.iter() {
                let dd = ((cx - wx).powi(2) + (cy - wy).powi(2)).sqrt();
                if dd < d {
                    d = dd;
                }
            }
            if d > bd {
                bd = d;
                best = (cx, cy);
            }
        }
        self.mx = best.0;
        self.my = best.1;

        // Rule 2 (VERBATIM): each wolf minimises  S = d(moose) - k*d(nearest OTHER wolf).
        // OPERATIONALIZED (Rust-forced, matching Python): wolves update SIMULTANEOUSLY —
        // every wolf reads the OLD pack positions (`self.wolves`, borrowed immutably) and
        // writes into a fresh `next` Vec, which replaces the pack only after the whole
        // phase. The borrow checker enforces the read/write split, so no wolf ever sees a
        // half-updated pack.
        let mut next = Vec::with_capacity(self.wolves.len());
        for (i, &(wx, wy)) in self.wolves.iter().enumerate() {
            let mut best = (wx, wy);
            let mut bs = 1e9f64;
            for (cx, cy) in Hunt::cands(wx, wy, self.vw) {
                if !self.in_bounds(cx, cy) {
                    continue;
                }
                let dm = ((cx - self.mx).powi(2) + (cy - self.my).powi(2)).sqrt();
                // nearest OTHER wolf; default 0.0 if none (single-wolf edge case).
                let mut do_ = f64::INFINITY;
                for (j, &(ox, oy)) in self.wolves.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    let dd = ((cx - ox).powi(2) + (cy - oy).powi(2)).sqrt();
                    if dd < do_ {
                        do_ = dd;
                    }
                }
                if !do_.is_finite() {
                    do_ = 0.0;
                }
                let s = dm - self.k * do_;
                if s < bs {
                    bs = s;
                    best = (cx, cy);
                }
            }
            next.push(best);
        }
        self.wolves = next;
        self.gap()
    }

    // Largest angular gap (deg) between adjacent wolves as seen from the moose.
    // 360/N when evenly ringed -> surrounded; near 360 when all on one side -> open escape.
    fn gap(&self) -> f64 {
        let mut angs: Vec<f64> = self
            .wolves
            .iter()
            .map(|&(wx, wy)| (wy - self.my).atan2(wx - self.mx))
            .collect();
        if angs.len() < 2 {
            return 360.0;
        }
        angs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = angs.len();
        let mut mx = 0.0f64;
        for i in 0..n {
            let mut g = (angs[(i + 1) % n] - angs[i]).rem_euclid(TAU);
            // rem_euclid gives [0,TAU); for i==n-1 the wrap is angs[0]-angs[n-1] (negative)
            // mod TAU, matching Python's %TAU.
            if g < 0.0 {
                g += TAU;
            }
            if g > mx {
                mx = g;
            }
        }
        mx * 180.0 / PI
    }

    fn nearest_wolf(&self) -> f64 {
        let mut md = f64::INFINITY;
        for &(wx, wy) in self.wolves.iter() {
            let d = ((self.mx - wx).powi(2) + (self.my - wy).powi(2)).sqrt();
            if d < md {
                md = d;
            }
        }
        md
    }
}

fn render(hunt: &Hunt) {
    let w = hunt.w as usize;
    let h = hunt.h as usize;
    let mut grid = vec![vec![' '; w]; h];
    for &(wx, wy) in hunt.wolves.iter() {
        let x = (wx as usize) % w;
        let y = (wy as usize) % h;
        grid[y][x] = 'W';
    }
    let mx = (hunt.mx as usize) % w;
    let my = (hunt.my as usize) % h;
    grid[my][mx] = 'M';
    println!("\nThe hunt (M moose, W wolves — watch the ring close):\n");
    for row in grid.iter() {
        let line: String = row.iter().collect();
        println!("{}", line);
    }
}

fn run(ticks: usize, n_wolves: usize, seed: u64, verbose: bool) -> (Hunt, Vec<f64>) {
    let mut hunt = Hunt::new(n_wolves, 80.0, 44.0, seed, 0.6, 1.0, 1.12);
    let mut hist = vec![hunt.gap()];
    let sample_every = std::cmp::max(1, ticks / 12);
    for t in 0..ticks {
        let g = hunt.step();
        if t % sample_every == 0 {
            hist.push(g);
        }
    }
    if verbose {
        render(&hunt);
        let md = hunt.nearest_wolf();
        println!(
            "\nlargest escape gap around the moose: {:.0}\u{00b0}  (evenly surrounded \u{2248} {}\u{00b0}) | nearest wolf {:.1}",
            hunt.gap(),
            360 / n_wolves,
            md
        );
        print!("gap\u{00b0}(t):");
        for g in hist.iter() {
            print!(" {:.0}", g);
        }
        println!();
    }
    (hunt, hist)
}

fn main() {
    let mut ticks: usize = 260;
    let mut n_wolves: usize = 6;
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
                ticks = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(260);
            }
            // Python names this flag --wolves; accept --ants too for cross-port CLI parity.
            "--wolves" | "--ants" => {
                i += 1;
                n_wolves = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(6);
            }
            _ => {}
        }
        i += 1;
    }

    run(ticks, n_wolves, seed, true);
}
