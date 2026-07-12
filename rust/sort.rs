// 'Go to the Ant' §3.2 — Ant Brood Sorting (Deneubourg et al. 1991), a faithful Rust port.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
// Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
//
// The RUST LENS: ownership and borrowing make "who is allowed to mutate the shared state
// this step" explicit and machine-checked. The brood grid lives in `Nest`; each ant borrows
// it mutably one at a time inside the tick loop, so the pick-up / put-down that edits the
// shared field is a single-owner mutation in a strictly serial order — the borrow checker
// guarantees no two ants alias the grid simultaneously. That serial discipline is exactly the
// Python semantics, made explicit rather than merely assumed.
//
// std-only, single file. Build:  rustc -O sort.rs   Run: ./sort --seed 0
//
// The four local rules (§3.2), preserved verbatim in spirit:
//   1. Wander randomly around the nest (dx,dy each in {-1,0,1}, toroidal).
//   2. Keep a SHORT memory (~10 steps) of the object types recently seen (incl. empties).
//   3. Not carrying + at an object: pick it up stochastically.  p(pickup) = (k+/(k+ + f))^2,
//      where f is the fraction of short-term memory occupied by the SAME type.
//      (Rare type -> f small -> pick up ~surely.)
//   4. Carrying + on empty ground: drop it stochastically.  p(putdown) = (f/(k- + f))^2.
//      (Surrounded by the same type -> f large -> drop ~surely.)
//   Constants (paper): k+ ~ 1, k- ~ 3 (k- must exceed k+ or clusters dissolve faster than
//   they form).  mem ~ 10.
// Local concentrations of like items emerge, retain members, and attract more; stochastic
// pickup lets separate clusters merge. Sorting EMERGES; no ant compares the whole nest.

const W: usize = 40;
const H: usize = 24;
const N_PER_TYPE: usize = 90;
const TYPES: [u8; 3] = [b'A', b'B', b'C'];

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
    fn randrange(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    // choice(list): list[randrange(len)]
    fn choice_i8(&mut self, list: &[i8]) -> i8 {
        list[self.randrange(list.len() as u64) as usize]
    }
    // shuffle(arr): Fisher-Yates, exactly matching the cross-port convention.
    fn shuffle<T>(&mut self, arr: &mut [T]) {
        for i in (1..arr.len()).rev() {
            let j = self.randrange((i + 1) as u64) as usize;
            arr.swap(i, j);
        }
    }
}

// A cell holds either no item (0) or a type byte b'A'/b'B'/b'C'.
struct Nest {
    grid: Vec<u8>, // W*H, row-major; 0 == empty
}

impl Nest {
    fn new(rng: &mut SplitMix64) -> Self {
        let mut grid = vec![0u8; W * H];
        // cells = [(x, y) for y in range(h) for x in range(w)] — same iteration order.
        let mut cells: Vec<(usize, usize)> = Vec::with_capacity(W * H);
        for y in 0..H {
            for x in 0..W {
                cells.push((x, y));
            }
        }
        rng.shuffle(&mut cells);
        let mut i = 0usize;
        for &t in TYPES.iter() {
            for _ in 0..N_PER_TYPE {
                let (x, y) = cells[i];
                i += 1;
                grid[y * W + x] = t;
            }
        }
        Nest { grid }
    }

    #[inline]
    fn at(&self, x: usize, y: usize) -> u8 {
        self.grid[y * W + x]
    }

    // Mean fraction of 8 (toroidal) neighbours that share an item's type.
    // 0 = scattered, 1 = perfectly sorted. Does not consume the RNG.
    fn clustering(&self) -> f64 {
        let mut tot = 0.0f64;
        let mut same = 0.0f64;
        for y in 0..H {
            for x in 0..W {
                let t = self.at(x, y);
                if t == 0 {
                    continue;
                }
                let mut neigh = 0.0f64;
                let mut simt = 0.0f64;
                for dx in [-1i32, 0, 1] {
                    for dy in [-1i32, 0, 1] {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = ((x as i32 + dx).rem_euclid(W as i32)) as usize;
                        let ny = ((y as i32 + dy).rem_euclid(H as i32)) as usize;
                        let nt = self.at(nx, ny);
                        if nt != 0 {
                            neigh += 1.0;
                            if nt == t {
                                simt += 1.0;
                            }
                        }
                    }
                }
                if neigh > 0.0 {
                    tot += 1.0;
                    same += simt / neigh;
                }
            }
        }
        same / tot.max(1.0)
    }
}

// Short-term memory ring buffer of the last `mem` seen types (incl. empties == 0).
struct SortAnt {
    x: usize,
    y: usize,
    carry: u8, // 0 == not carrying
    mem: Vec<u8>,
    mem_cap: usize,
    kp: f64,
    km: f64,
}

impl SortAnt {
    fn new(nest_rng: &mut SplitMix64, mem: usize, kp: f64, km: f64) -> Self {
        // Matches Python: x = randrange(w); y = randrange(h) — in this order, per ant.
        let x = nest_rng.randrange(W as u64) as usize;
        let y = nest_rng.randrange(H as u64) as usize;
        SortAnt {
            x,
            y,
            carry: 0,
            mem: Vec::with_capacity(mem),
            mem_cap: mem,
            kp,
            km,
        }
    }

    // f = fraction of short-term memory holding the SAME type t.
    fn f(&self, t: u8) -> f64 {
        if self.mem.is_empty() {
            return 0.0;
        }
        let c = self.mem.iter().filter(|&&m| m == t).count();
        c as f64 / self.mem.len() as f64
    }

    fn remember(&mut self, here: u8) {
        // deque(maxlen=mem): append, evicting the oldest when full.
        if self.mem.len() == self.mem_cap {
            self.mem.remove(0);
        }
        self.mem.push(here);
    }

    fn step(&mut self, nest: &mut Nest, rng: &mut SplitMix64) {
        // rule 1: wander — dx then dy, each choice((-1,0,1)), toroidal.
        let dx = rng.choice_i8(&[-1, 0, 1]) as i32;
        self.x = ((self.x as i32 + dx).rem_euclid(W as i32)) as usize;
        let dy = rng.choice_i8(&[-1, 0, 1]) as i32;
        self.y = ((self.y as i32 + dy).rem_euclid(H as i32)) as usize;

        let here = nest.at(self.x, self.y);
        self.remember(here); // rule 2 (record even empties)

        if self.carry == 0 {
            if here != 0 {
                // rule 3: maybe pick up.
                // PAPER §3.2 VERBATIM: p(pickup) = (k+/(k+ + f))^2
                let p = (self.kp / (self.kp + self.f(here))).powi(2);
                if rng.random_float() < p {
                    self.carry = here;
                    nest.grid[self.y * W + self.x] = 0;
                }
            }
        } else {
            if here == 0 {
                // rule 4: maybe drop.
                // PAPER §3.2 VERBATIM: p(putdown) = (f/(k- + f))^2
                let fc = self.f(self.carry);
                let p = (fc / (self.km + fc)).powi(2);
                if rng.random_float() < p {
                    nest.grid[self.y * W + self.x] = self.carry;
                    self.carry = 0;
                }
            }
        }
    }
}

fn render(nest: &Nest) {
    for y in 0..H {
        let mut line = String::with_capacity(W);
        for x in 0..W {
            let c = nest.at(x, y);
            line.push(if c == 0 { '.' } else { c as char });
        }
        println!("{}", line);
    }
}

fn run(ticks: usize, n_ants: usize, seed: u64, verbose: bool) -> (Nest, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let mut nest = Nest::new(&mut rng);
    // ants = [SortAnt(nest) for _ in range(n_ants)] — construction consumes RNG in order.
    let mut ants: Vec<SortAnt> = (0..n_ants)
        .map(|_| SortAnt::new(&mut rng, 10, 1.0, 3.0))
        .collect();

    if verbose {
        println!("BEFORE (random scatter):\n");
        render(&nest);
        println!("\ninitial clustering: {:.3}", nest.clustering());
    }

    let mut hist: Vec<f64> = Vec::new();
    let sample_every = std::cmp::max(1, ticks / 12);
    for t in 0..ticks {
        // RUST LENS: one ant at a time takes a &mut borrow of the shared nest — the serial
        // update order is enforced by the borrow checker, not left implicit.
        for a in ants.iter_mut() {
            a.step(&mut nest, &mut rng);
        }
        if t % sample_every == 0 {
            hist.push(nest.clustering());
        }
    }

    if verbose {
        println!("\nAFTER (emergent sorting):\n");
        render(&nest);
        println!("\nfinal clustering: {:.3}", nest.clustering());
        print!("clustering(t):");
        for c in hist.iter() {
            print!(" {:.2}", c);
        }
        println!();
    }
    (nest, hist)
}

fn main() {
    let mut ticks: usize = 120000;
    let mut ants: usize = 40;
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
                ticks = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(120000);
            }
            "--ants" => {
                i += 1;
                ants = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(40);
            }
            _ => {}
        }
        i += 1;
    }

    run(ticks, ants, seed, true);
}
