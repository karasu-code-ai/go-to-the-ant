// 'Go to the Ant' — a faithful Rust port of Parunak's OG foraging swarm.
//
// Source: H. Van Dyke Parunak, "'Go to the Ant': Engineering Principles from Natural
// Multi-Agent Systems," Annals of Operations Research 75:69-101 (1997).
//
// The RUST LENS: ownership answers "who owns the shared field?". In this port the two
// pheromone fields live in `World`; ants borrow it mutably one at a time inside the tick
// loop, so the "communication through the environment" channel is a single-owner resource
// mutated in a strictly serial order — the borrow checker guarantees no two ants alias the
// field simultaneously. That serial discipline is exactly the Python semantics, made
// explicit and machine-checked rather than assumed.
//
// std-only, single file. Build:  rustc -O forage.rs   Run: ./forage --seed 0
//
// The five local ant rules (§3.1 "Ants: Path planning"), preserved verbatim in spirit:
//   1. Avoid obstacles.
//   2. Wander randomly, biased toward nearby pheromone (Brownian floor + local scent bias).
//   3. If holding food, drop pheromone at a CONSTANT RATE while walking.
//   4. If at food and not holding any, pick it up.
//   5. If at the nest and carrying food, drop it.
// Plus the field law: pheromone EVAPORATES every tick (the entropy leak), so trails laid by
// ants who never got home — and paths to depleted sources — fade. No ant plans a route; the
// network EMERGES from deposit + evaporation + weighted-random following.
//
// PRINCIPLE APPLIED (§4.3.3 Small in Scope + §4.6 multi-marker): TWO local pheromone fields,
// not a global homing beacon. Searchers follow FOOD scent; carriers follow the HOME gradient
// diffused outward from the nest. No ant knows where the nest is — it climbs the local
// gradient only.

const W: usize = 56;
const H: usize = 28;
const REGION: i32 = 2;

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
}

struct World {
    food_pher: Vec<f64>, // laid by carriers; followed by searchers
    home_pher: Vec<f64>, // laid/diffused by the nest; followed by carriers
    obstacle: Vec<bool>,
    nest: (i32, i32),
    food: (i32, i32),
    food_qty: i64,
    deliveries: i64,
}

impl World {
    fn new(use_wall: bool, rng: &mut SplitMix64) -> Self {
        let mut obstacle = vec![false; W * H];
        let nest = (5, (H / 2) as i32);
        let food = ((W - 6) as i32, (H / 2) as i32);
        if use_wall {
            // Optional wall with a single gap -> the swarm must ROUTE around it.
            let wallx = (W / 2) as i32;
            // randint(4, H-5) inclusive; operationalized with SplitMix64.
            let lo = 4i64;
            let hi = (H - 5) as i64;
            let span = (hi - lo + 1) as u64;
            let gap = lo + (rng.next_u64() % span) as i64;
            for y in 0..H as i32 {
                if (y as i64 - gap).abs() > 2 {
                    obstacle[y as usize * W + wallx as usize] = true;
                }
            }
        }
        World {
            food_pher: vec![0.0; W * H],
            home_pher: vec![0.0; W * H],
            obstacle,
            nest,
            food,
            food_qty: 1_000_000_000,
            deliveries: 0,
        }
    }

    #[inline]
    fn idx(x: i32, y: i32) -> usize {
        y as usize * W + x as usize
    }

    #[inline]
    fn free(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < W as i32 && y >= 0 && y < H as i32 && !self.obstacle[Self::idx(x, y)]
    }

    fn at_food(&self, x: i32, y: i32) -> bool {
        (x - self.food.0).abs() <= REGION && (y - self.food.1).abs() <= REGION
    }

    fn at_nest(&self, x: i32, y: i32) -> bool {
        (x - self.nest.0).abs() <= REGION && (y - self.nest.1).abs() <= REGION
    }

    fn evaporate(&mut self, rate: f64) {
        // Both fields dissipate (the entropy leak).
        let keep = 1.0 - rate;
        for i in 0..W * H {
            self.food_pher[i] *= keep;
            self.home_pher[i] *= keep;
        }
    }

    fn emit_and_diffuse_home(&mut self) {
        // The NEST is a home-pheromone SOURCE; the marker DIFFUSES outward into a gradient
        // that points home from everywhere. Carriers read only the LOCAL gradient.
        let (nx, ny) = self.nest;
        for dy in -REGION..=REGION {
            for dx in -REGION..=REGION {
                let (x, y) = (nx + dx, ny + dy);
                if self.free(x, y) {
                    self.home_pher[Self::idx(x, y)] += 6.0;
                }
            }
        }
        // One simultaneous (Jacobi) diffusion pass over a copy.
        let mut nxt = self.home_pher.clone();
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                if !self.free(x, y) {
                    continue;
                }
                let mut s = self.home_pher[Self::idx(x, y)];
                let mut c = 1.0f64;
                for &(dx, dy) in DIRS.iter() {
                    let (xx, yy) = (x + dx, y + dy);
                    if self.free(xx, yy) {
                        s += self.home_pher[Self::idx(xx, yy)];
                        c += 1.0;
                    }
                }
                nxt[Self::idx(x, y)] = s / c;
            }
        }
        self.home_pher = nxt;
    }
}

struct Ant {
    x: i32,
    y: i32,
    carrying: bool,
}

impl Ant {
    fn new(world: &World) -> Self {
        Ant {
            x: world.nest.0,
            y: world.nest.1,
            carrying: false,
        }
    }

    fn step(&mut self, world: &mut World, deposit: f64, rng: &mut SplitMix64) {
        // Rule 2, fully LOCAL: follow the field that leads where you're going. Searchers read
        // the FOOD scent; carriers read the HOME scent. Brownian floor keeps the walk alive.
        let mut wts = [0.0f64; 8];
        let mut tot = 0.0f64;
        for (i, &(dx, dy)) in DIRS.iter().enumerate() {
            let (nx, ny) = (self.x + dx, self.y + dy);
            if !world.free(nx, ny) {
                wts[i] = 0.0; // rule 1: never step into a wall
                continue;
            }
            let field = if self.carrying {
                world.home_pher[World::idx(nx, ny)]
            } else {
                world.food_pher[World::idx(nx, ny)]
            };
            wts[i] = 1.0 + field * 6.0; // Brownian floor + local scent bias
            tot += wts[i];
        }
        if tot <= 0.0 {
            return; // boxed in — stay put this tick
        }
        let r = rng.random_float() * tot;
        let mut acc = 0.0f64;
        for (i, &(dx, dy)) in DIRS.iter().enumerate() {
            acc += wts[i];
            if r <= acc {
                // Note the <=, matching the Python.
                self.x += dx;
                self.y += dy;
                break;
            }
        }
        // Rule 3: carriers lay the FOOD trail (the nest broadcasts the HOME field).
        if self.carrying {
            world.food_pher[World::idx(self.x, self.y)] += deposit;
        }
        // Rule 4: pick up food.
        if world.at_food(self.x, self.y) && !self.carrying && world.food_qty > 0 {
            self.carrying = true;
            world.food_qty -= 1;
        } else if world.at_nest(self.x, self.y) && self.carrying {
            // Rule 5: drop food at the nest.
            self.carrying = false;
            world.deliveries += 1;
        }
    }
}

fn run(
    ticks: usize,
    n_ants: usize,
    evap: f64,
    deposit: f64,
    seed: u64,
    use_wall: bool,
) -> (World, Vec<Ant>, Vec<(usize, i64)>) {
    let mut rng = SplitMix64::new(seed);
    let mut world = World::new(use_wall, &mut rng);
    let mut ants: Vec<Ant> = (0..n_ants).map(|_| Ant::new(&world)).collect();
    let mut history: Vec<(usize, i64)> = Vec::new();
    let sample_every = std::cmp::max(1, ticks / 20);
    for t in 0..ticks {
        world.emit_and_diffuse_home(); // nest broadcasts the home gradient
        for a in ants.iter_mut() {
            a.step(&mut world, deposit, &mut rng);
        }
        world.evaporate(evap);
        if t % sample_every == 0 {
            history.push((t, world.deliveries));
        }
    }
    (world, ants, history)
}

fn render_ascii(world: &World, ants: &[Ant]) {
    let mut peak = 0.0f64;
    for &v in world.food_pher.iter() {
        if v > peak {
            peak = v;
        }
    }
    if peak == 0.0 {
        peak = 1.0;
    }
    let shades: Vec<char> = " .:-=+*#%@".chars().collect();
    let n_shades = shades.len();
    let antpos: std::collections::HashSet<(i32, i32)> =
        ants.iter().map(|a| (a.x, a.y)).collect();
    println!(
        "\nGo to the Ant — emergent trail (nest N <-> food F, wall '|', pheromone density):\n"
    );
    for y in 0..H as i32 {
        let mut line = String::with_capacity(W);
        for x in 0..W as i32 {
            let c = if (x, y) == world.nest {
                'N'
            } else if (x, y) == world.food {
                'F'
            } else if world.obstacle[World::idx(x, y)] {
                '|'
            } else if antpos.contains(&(x, y)) {
                'o'
            } else {
                let lvl = ((world.food_pher[World::idx(x, y)] / peak) * (n_shades - 1) as f64)
                    as i32;
                let lvl = lvl.max(0).min(n_shades as i32 - 1) as usize;
                shades[lvl]
            };
            line.push(c);
        }
        println!("{}", line);
    }
}

fn main() {
    let mut ticks: usize = 3000;
    let mut ants: usize = 90;
    let mut evap: f64 = 0.015;
    let deposit: f64 = 1.0;
    let mut seed: u64 = 0;
    let mut use_wall = false;

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
                ticks = args[i].parse().unwrap_or(3000);
            }
            "--ants" => {
                i += 1;
                ants = args[i].parse().unwrap_or(90);
            }
            "--evap" => {
                i += 1;
                evap = args[i].parse().unwrap_or(0.015);
            }
            "--wall" => {
                use_wall = true;
            }
            _ => {}
        }
        i += 1;
    }

    let (world, final_ants, history) = run(ticks, ants, evap, deposit, seed, use_wall);
    render_ascii(&world, &final_ants);
    println!(
        "\nfood delivered to nest over {} ticks: {}",
        ticks, world.deliveries
    );
    print!("deliveries(t):");
    for (_, d) in history.iter() {
        print!(" {}", d);
    }
    println!();
}
