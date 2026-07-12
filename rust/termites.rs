// 'Go to the Ant' §3.3 — Termite Nest Building (Kugler et al. 1990), a faithful Rust port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
// Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
//
// Tropical termites raise 5-metre mounds — columns, arches, floors — with no chief engineer.
// Parunak's three local rules (§3.3):
//   1. Metabolize bodily waste, which contains pheromone. The waste IS the building material.
//   2. Wander randomly, but prefer the direction of the strongest local pheromone concentration.
//   3. Each step, decide stochastically whether to deposit the current load. p(deposit) rises
//      with the LOCAL pheromone density AND the amount carried. A full termite drops even with
//      no nearby deposit; a termite in a very high local concentration drops even a small load.
// Because pheromone DECAYS, the freshest deposits (the centre of a growing pile) smell strongest,
// so piles climb upward into COLUMNS rather than spreading. No termite plans the mound.
//
// Here (top-down 2D): `mass` = persistent structural mass (what you see); `scent` = decaying
// pheromone (what biases wandering). Emergence = scattered dabs self-concentrate into a handful
// of tall columns.
//
// The RUST LENS: ownership makes "who is allowed to mutate the shared state this step" explicit
// and machine-checked. The two fields (mass, scent) live in one `Mound`; termites borrow it
// mutably one at a time inside the tick loop, so the stigmergic channel is a single-owner
// resource mutated in strictly serial order. The borrow checker guarantees no two termites alias
// the field simultaneously — exactly the Python semantics, made explicit rather than assumed.
//
// std-only, single file. Build:  rustc -O termites.rs   Run: ./termites --seed 0

// The 8 neighbours in THIS order (must match the Python exactly).
const DIRS: [(i32, i32); 8] = [
    (-1, -1), (0, -1), (1, -1), (-1, 0),
    (1, 0), (-1, 1), (0, 1), (1, 1),
];

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
    // randrange(n) -> [0, n)
    fn randrange(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

struct Mound {
    w: usize,
    h: usize,
    mass: Vec<f64>,  // persistent structure (viz)
    scent: Vec<f64>, // decaying pheromone (bias)
    decay: f64,
}

impl Mound {
    fn new(w: usize, h: usize, decay: f64) -> Self {
        Mound {
            w,
            h,
            mass: vec![0.0; w * h],
            scent: vec![0.0; w * h],
            decay,
        }
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }

    fn evaporate(&mut self) {
        let keep = 1.0 - self.decay;
        for v in self.scent.iter_mut() {
            *v *= keep;
        }
    }

    // Count distinct COLUMNS = toroidal local maxima with mass above a fraction of the tallest peak.
    fn columns(&self) -> (usize, f64) {
        let mut peak = 0.0f64;
        for &v in self.mass.iter() {
            if v > peak {
                peak = v;
            }
        }
        if peak <= 0.0 {
            return (0, 0.0);
        }
        let cut = peak * 0.15;
        let mut cnt = 0usize;
        for y in 0..self.h {
            for x in 0..self.w {
                let v = self.mass[self.idx(x, y)];
                if v < cut {
                    continue;
                }
                let mut is_max = true;
                for &(dx, dy) in DIRS.iter() {
                    let nx = ((x as i32 + dx).rem_euclid(self.w as i32)) as usize;
                    let ny = ((y as i32 + dy).rem_euclid(self.h as i32)) as usize;
                    if v < self.mass[self.idx(nx, ny)] {
                        is_max = false;
                        break;
                    }
                }
                if is_max {
                    cnt += 1;
                }
            }
        }
        (cnt, peak)
    }
}

struct Termite {
    x: usize,
    y: usize,
    load: f64,
    metab: f64,
    maxload: f64,
}

impl Termite {
    fn new(mound: &Mound, metab: f64, maxload: f64, rng: &mut SplitMix64) -> Self {
        // NOTE: Python constructs x=randrange(w) then y=randrange(h) — same order here.
        let x = rng.randrange(mound.w as u64) as usize;
        let y = rng.randrange(mound.h as u64) as usize;
        Termite {
            x,
            y,
            load: 0.0,
            metab,
            maxload,
        }
    }

    fn step(&mut self, mound: &mut Mound, rng: &mut SplitMix64) {
        // rule 1: metabolize -> waste accumulates
        self.load = self.maxload.min(self.load + self.metab);
        // rule 2: wander, biased toward the strongest local scent
        let mut wts = [0.0f64; 8];
        let mut nbr = [(0usize, 0usize); 8];
        let mut tot = 0.0f64;
        for (i, &(dx, dy)) in DIRS.iter().enumerate() {
            let nx = ((self.x as i32 + dx).rem_euclid(mound.w as i32)) as usize;
            let ny = ((self.y as i32 + dy).rem_euclid(mound.h as i32)) as usize;
            let w = 1.0 + mound.scent[mound.idx(nx, ny)] * 3.0;
            wts[i] = w;
            nbr[i] = (nx, ny);
            tot += w;
        }
        let mut r = rng.random_float() * tot;
        for i in 0..8 {
            r -= wts[i];
            if r <= 0.0 {
                self.x = nbr[i].0;
                self.y = nbr[i].1;
                break;
            }
        }
        // rule 3: stochastic deposit — rises with local scent AND load; full termite always drops
        let local = mound.scent[mound.idx(self.x, self.y)];
        // OPERATIONALIZED: paper §3.3 gives NO formula, only "prob rises with local density AND load"
        let p = 1.0f64.min(0.01 + 0.55 * (self.load / self.maxload) + 0.20 * local);
        if self.load >= self.maxload || rng.random_float() < p {
            let i = mound.idx(self.x, self.y);
            mound.mass[i] += self.load;
            mound.scent[i] += self.load;
            self.load = 0.0;
        }
    }
}

fn run(
    ticks: usize,
    n: usize,
    seed: u64,
    decay: f64,
    verbose: bool,
) -> (Mound, Vec<usize>) {
    let (w, h) = (58usize, 34usize);
    let mut rng = SplitMix64::new(seed);
    let mut mound = Mound::new(w, h, decay);
    let mut termites: Vec<Termite> = (0..n)
        .map(|_| Termite::new(&mound, 0.4, 6.0, &mut rng))
        .collect();
    let mut hist: Vec<usize> = Vec::new();
    let sample_every = std::cmp::max(1, ticks / 12);
    for t in 0..ticks {
        for tm in termites.iter_mut() {
            tm.step(&mut mound, &mut rng);
        }
        mound.evaporate();
        if t % sample_every == 0 {
            hist.push(mound.columns().0);
        }
    }
    if verbose {
        render(&mound);
        let (cnt, peak) = mound.columns();
        println!(
            "\ndistinct columns (local maxima): {} | tallest column mass: {:.0}",
            cnt, peak
        );
        print!("columns(t):");
        for c in hist.iter() {
            print!(" {}", c);
        }
        println!();
    }
    (mound, hist)
}

fn render(mound: &Mound) {
    let mut peak = 0.0f64;
    for &v in mound.mass.iter() {
        if v > peak {
            peak = v;
        }
    }
    if peak <= 0.0 {
        peak = 1.0;
    }
    let shades: Vec<char> = " .:-=+*#%@".chars().collect();
    let n_shades = shades.len();
    println!(
        "\nTermite mound (top-down mass density — columns emerge as bright cores):\n"
    );
    for y in 0..mound.h {
        let mut line = String::with_capacity(mound.w);
        for x in 0..mound.w {
            let v = mound.mass[mound.idx(x, y)];
            let lvl = (v / peak * (n_shades - 1) as f64) as i32;
            let lvl = lvl.max(0).min(n_shades as i32 - 1) as usize;
            line.push(shades[lvl]);
        }
        println!("{}", line);
    }
}

fn main() {
    let mut ticks: usize = 40000;
    let mut n: usize = 70;
    let mut decay: f64 = 0.02;
    let mut seed: u64 = 0;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args[i].parse().unwrap_or(0);
            }
            "--ticks" => {
                i += 1;
                ticks = args[i].parse().unwrap_or(40000);
            }
            // Python names the flag --termites; accept --ants too for cross-port CLI parity.
            "--termites" | "--ants" => {
                i += 1;
                n = args[i].parse().unwrap_or(70);
            }
            "--decay" => {
                i += 1;
                decay = args[i].parse().unwrap_or(0.02);
            }
            _ => {}
        }
        i += 1;
    }

    run(ticks, n, seed, decay, true);
}
