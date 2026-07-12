// 'Go to the Ant' §3.5 — Birds & Fish: Flocking (Reynolds 1987, Heppner 1990),
// a faithful Rust port recreated from the paper.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
// Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
//
// Flocks stay together, turn together, and avoid collisions with no leader and no central
// coordinator — each bird senses only its nearest peers. Parunak lists Reynolds' three local
// rules (§3.5), which are the paper's:
//   1. SEPARATION — keep a minimum distance from the nearest birds (avoid collisions).
//   2. ALIGNMENT  — match velocity (speed + heading) to nearby birds.
//   3. COHESION   — stay close to the centre of the local flock.
// Each rule is a steering vector from the neighbours inside a perception radius; their sum turns
// the bird. Global coordination (a single coherent, banking flock) EMERGES from these three
// local urges.
//
// PROVENANCE: the three rules are the paper's (Reynolds' "boids"). The perception radius, the
// separation distance, and the three weights are OPERATIONALIZED — Parunak's paper lists the
// rules but gives no numbers (Reynolds 1987 is the primary source for tuned constants).
//
// The RUST LENS: ownership and borrowing make "who is allowed to mutate the shared state this
// step" explicit and machine-checked. The whole flock state (positions + velocities) lives in
// one `Flock`. `step` computes every bird's NEXT velocity into freshly-owned `nvx/nvy` buffers
// while holding only shared (`&`) reads of the current state, then commits them in a second
// pass — so the double-buffered "all birds see the same snapshot this tick" semantics are not a
// convention you must remember but a fact the borrow checker enforces: no bird can mutate the
// shared field another bird is still reading.
//
// std-only, single file. Build:  rustc -O flocking.rs    Run: ./flocking --seed 0
//
// NO per-step randomness — fully deterministic given the random init, so cross-port identity
// depends only on matching the init-RNG order (all px, then all py, then all headings) and the
// neighbour-sum order (iterate j in index order).

use std::f64::consts::PI;

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
    // uniform(a,b): a + random_float()*(b-a)
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + self.random_float() * (b - a)
    }
}

struct Flock {
    n: usize,
    w: f64,
    h: f64,
    px: Vec<f64>,
    py: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    perc: f64,
    sep_r: f64,
    wsep: f64,
    wali: f64,
    wcoh: f64,
    vmax: f64,
    turn: f64,
}

impl Flock {
    fn new(n: usize, w: f64, h: f64, seed: u64) -> Self {
        // OPERATIONALIZED constants (Reynolds 1987 supplies the tuned numbers Parunak omits).
        let perc = 8.0;
        let sep_r = 3.0;
        let wsep = 1.3;
        let wali = 1.5;
        let wcoh = 0.85;
        let vmax = 1.0;
        let turn = 0.35;
        let mut rng = SplitMix64::new(seed);
        // Init consumes the RNG in EXACTLY the Python order: all px, then all py, then all headings.
        let px: Vec<f64> = (0..n).map(|_| rng.uniform(0.0, w)).collect();
        let py: Vec<f64> = (0..n).map(|_| rng.uniform(0.0, h)).collect();
        let ang: Vec<f64> = (0..n).map(|_| rng.uniform(0.0, 2.0 * PI)).collect();
        let vx: Vec<f64> = ang.iter().map(|a| a.cos()).collect();
        let vy: Vec<f64> = ang.iter().map(|a| a.sin()).collect();
        Flock {
            n,
            w,
            h,
            px,
            py,
            vx,
            vy,
            perc,
            sep_r,
            wsep,
            wali,
            wcoh,
            vmax,
            turn,
        }
    }

    fn step(&mut self) -> f64 {
        let n = self.n;
        // Double buffer: this step's NEXT velocities. Every bird reads the SAME current-state
        // snapshot (shared &self borrow) and writes only into these owned buffers — the borrow
        // checker guarantees no bird mutates state another is still reading.
        let mut nvx = self.vx.clone();
        let mut nvy = self.vy.clone();
        let p2 = self.perc * self.perc;
        let s2 = self.sep_r * self.sep_r;
        for i in 0..n {
            let (mut sx, mut sy) = (0.0f64, 0.0f64);
            let (mut ax, mut ay) = (0.0f64, 0.0f64);
            let (mut cx, mut cy) = (0.0f64, 0.0f64);
            let mut cnt = 0i64;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut dx = self.px[j] - self.px[i];
                let mut dy = self.py[j] - self.py[i];
                // toroidal delta (nearest image); .round() = half-away-from-zero, matching the
                // other sequential ports (Go/C). Not Python's half-to-even, but .5 hits are
                // effectively impossible on these float positions and all ports agree here.
                dx -= self.w * (dx / self.w).round();
                dy -= self.h * (dy / self.h).round();
                let d2 = dx * dx + dy * dy;
                if d2 > p2 {
                    continue;
                }
                cnt += 1;
                ax += self.vx[j]; // rule 2: alignment (avg neighbour velocity)
                ay += self.vy[j];
                cx += dx; // rule 3: cohesion (toward neighbour centre)
                cy += dy;
                if d2 < s2 && d2 > 1e-9 {
                    // rule 1: separation (push from the close ones)
                    sx -= dx / d2;
                    sy -= dy / d2;
                }
            }
            if cnt > 0 {
                let cf = cnt as f64;
                ax /= cf;
                ay /= cf;
                cx /= cf;
                cy /= cf;
                // NORMALIZE each urge to a unit vector so the three weights are actually
                // comparable (otherwise the position-scale cohesion vector swamps the
                // velocity-scale alignment one).
                let u = |x: f64, y: f64| -> (f64, f64) {
                    let m = x.hypot(y);
                    if m > 1e-9 {
                        (x / m, y / m)
                    } else {
                        (0.0, 0.0)
                    }
                };
                let (sux, suy) = u(sx, sy); // rule 1: away from close birds
                let (aux, auy) = u(ax - self.vx[i], ay - self.vy[i]); // rule 2: toward neighbours' heading
                let (cux, cuy) = u(cx, cy); // rule 3: toward neighbours' centre
                let accx = self.wsep * sux + self.wali * aux + self.wcoh * cux;
                let accy = self.wsep * suy + self.wali * auy + self.wcoh * cuy;
                let mut vxi = self.vx[i] + self.turn * accx;
                let mut vyi = self.vy[i] + self.turn * accy;
                let mut sp = vxi.hypot(vyi); // cap speed to vmax
                if sp == 0.0 {
                    sp = 1.0;
                }
                vxi = vxi / sp * self.vmax;
                vyi = vyi / sp * self.vmax;
                nvx[i] = vxi;
                nvy[i] = vyi;
            }
        }
        // Commit: adopt the new velocities and advance positions toroidally.
        for i in 0..n {
            self.vx[i] = nvx[i];
            self.vy[i] = nvy[i];
            self.px[i] = (self.px[i] + self.vx[i]).rem_euclid(self.w);
            self.py[i] = (self.py[i] + self.vy[i]).rem_euclid(self.h);
        }
        self.polarization()
    }

    // Order parameter: |mean heading|, 0 = disordered, 1 = one coherent flock.
    fn polarization(&self) -> f64 {
        let mx: f64 = self.vx.iter().sum::<f64>() / self.n as f64;
        let my: f64 = self.vy.iter().sum::<f64>() / self.n as f64;
        mx.hypot(my) / self.vmax
    }
}

fn render(fl: &Flock) {
    let w = fl.w as usize;
    let h = fl.h as usize;
    let mut grid = vec![vec![' '; w]; h];
    let arrow: Vec<char> = "→↗↑↖←↙↓↘".chars().collect();
    for i in 0..fl.n {
        let x = (fl.px[i] as usize) % w;
        let y = (fl.py[i] as usize) % h;
        let a = fl.vy[i].atan2(fl.vx[i]);
        let mut k = (a / (PI / 4.0)).round() as i64 % 8;
        if k < 0 {
            k += 8;
        }
        grid[y][x] = arrow[k as usize];
    }
    println!("\nFlock (each bird points along its heading — watch them align):\n");
    for row in grid.iter() {
        let line: String = row.iter().collect();
        println!("{}", line);
    }
}

fn run(ticks: usize, n: usize, seed: u64) {
    let mut fl = Flock::new(n, 90.0, 48.0, seed);
    let mut hist: Vec<f64> = vec![fl.polarization()];
    let sample_every = std::cmp::max(1, ticks / 12);
    for t in 0..ticks {
        let p = fl.step();
        if t % sample_every == 0 {
            hist.push(p);
        }
    }
    render(&fl);
    println!(
        "\npolarization (flock alignment): {:.3}  (0 = chaos, 1 = one flock)",
        fl.polarization()
    );
    print!("polarization(t):");
    for c in hist.iter() {
        print!(" {:.2}", c);
    }
    println!();
}

fn main() {
    let mut ticks: usize = 600;
    let mut birds: usize = 90;
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
                ticks = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(600);
            }
            // Python names this --birds; accept --ants too for cross-port CLI consistency.
            "--birds" | "--ants" => {
                i += 1;
                birds = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(90);
            }
            _ => {}
        }
        i += 1;
    }

    run(ticks, birds, seed);
}
