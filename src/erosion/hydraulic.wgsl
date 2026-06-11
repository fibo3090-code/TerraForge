// Hydraulic erosion, pipe model (Mei et al. 2007).
// Seven passes per iteration, dispatched in order:
//   rain -> compute_flux -> water_velocity -> compute_capacity
//        -> erode_deposit -> advect_sediment -> finalize_iter
// Every thread writes only its own cell in every pass, so the simulation is
// race-free and deterministic. Cell size l = 1; pipe cross-section A and
// gravity g are folded into flux_factor = dt * A * g / l on the CPU.

struct Params {
    width: u32,
    height: u32,
    dt: f32,
    rain_rate: f32,
    evaporation: f32,
    capacity_k: f32,
    erosion_k: f32,
    deposition_k: f32,
    min_tilt: f32,
    flux_factor: f32,
    dig_cap: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> terrain: array<f32>;
@group(0) @binding(2) var<storage, read_write> water: array<f32>;
@group(0) @binding(3) var<storage, read_write> sediment: array<f32>;
@group(0) @binding(4) var<storage, read_write> sediment_new: array<f32>;
// flux components: .x -> (x-1), .y -> (x+1), .z -> (z-1), .w -> (z+1)
@group(0) @binding(5) var<storage, read_write> flux: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> velocity: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> capacity: array<f32>;

fn idx(x: u32, z: u32) -> u32 {
    return z * p.width + x;
}

fn surface_height(x: u32, z: u32) -> f32 {
    let i = idx(x, z);
    return terrain[i] + water[i];
}

fn in_bounds(gid: vec3<u32>) -> bool {
    return gid.x < p.width && gid.y < p.height;
}

// Pass 1: rainfall increments water uniformly.
@compute @workgroup_size(8, 8, 1)
fn rain(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let i = idx(gid.x, gid.y);
    water[i] = water[i] + p.rain_rate * p.dt;
}

// Pass 2: outflow flux through the four virtual pipes, scaled so a cell
// never exports more water than it holds.
@compute @workgroup_size(8, 8, 1)
fn compute_flux(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let x = gid.x;
    let z = gid.y;
    let i = idx(x, z);
    let h0 = surface_height(x, z);
    let f = flux[i];

    var fl = 0.0;
    var fr = 0.0;
    var fu = 0.0;
    var fd = 0.0;
    if (x > 0u)            { fl = max(0.0, f.x + p.flux_factor * (h0 - surface_height(x - 1u, z))); }
    if (x + 1u < p.width)  { fr = max(0.0, f.y + p.flux_factor * (h0 - surface_height(x + 1u, z))); }
    if (z > 0u)            { fu = max(0.0, f.z + p.flux_factor * (h0 - surface_height(x, z - 1u))); }
    if (z + 1u < p.height) { fd = max(0.0, f.w + p.flux_factor * (h0 - surface_height(x, z + 1u))); }

    let total = fl + fr + fu + fd;
    if (total > 0.0) {
        // Cell area is 1, so available volume == water depth.
        let k = min(1.0, water[i] / (total * p.dt + 1e-7));
        fl = fl * k;
        fr = fr * k;
        fu = fu * k;
        fd = fd * k;
    }
    flux[i] = vec4<f32>(fl, fr, fu, fd);
}

// Pass 3: update water depth from net flux; derive the velocity field from
// the mean flow through the cell divided by mean water depth.
@compute @workgroup_size(8, 8, 1)
fn water_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let x = gid.x;
    let z = gid.y;
    let i = idx(x, z);
    let f = flux[i];
    let out_total = f.x + f.y + f.z + f.w;

    var left_in = 0.0;
    var right_in = 0.0;
    var up_in = 0.0;
    var down_in = 0.0;
    if (x > 0u)            { left_in  = flux[idx(x - 1u, z)].y; }
    if (x + 1u < p.width)  { right_in = flux[idx(x + 1u, z)].x; }
    if (z > 0u)            { up_in    = flux[idx(x, z - 1u)].w; }
    if (z + 1u < p.height) { down_in  = flux[idx(x, z + 1u)].z; }
    let inflow = left_in + right_in + up_in + down_in;

    let d_old = water[i];
    let d_new = max(0.0, d_old + p.dt * (inflow - out_total));
    water[i] = d_new;
    // Publish the flux-form advection result (advect_sediment ran just
    // before this pass and only wrote sediment_new).
    sediment[i] = sediment_new[i];

    let wx = (left_in - f.x + f.y - right_in) * 0.5;
    let wz = (up_in - f.z + f.w - down_in) * 0.5;
    let d_avg = max(0.5 * (d_old + d_new), 0.01);
    var v = vec2<f32>(wx / d_avg, wz / d_avg);
    // Clamp speed: the depth clamp above makes nearly-dry cells report
    // unphysically fast flow, and flux inertia lets pooled water slosh;
    // either would blow up sediment capacity. Physical flow on normalized
    // [0,1] terrain is O(0.1..2) cells per time unit.
    let speed = length(v);
    const MAX_SPEED: f32 = 2.0;
    if (speed > MAX_SPEED) {
        v = v * (MAX_SPEED / speed);
    }
    velocity[i] = v;
}

// Pass 4: sediment transport capacity from local tilt and flow speed.
// Reads neighbour terrain (read-only this pass) so erode_deposit can stay
// cell-local and race-free.
@compute @workgroup_size(8, 8, 1)
fn compute_capacity(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let x = gid.x;
    let z = gid.y;
    let i = idx(x, z);

    let xl = max(x, 1u) - 1u;
    let xr = min(x + 1u, p.width - 1u);
    let zu = max(z, 1u) - 1u;
    let zd = min(z + 1u, p.height - 1u);
    let dhdx = (terrain[idx(xr, z)] - terrain[idx(xl, z)]) / max(f32(xr - xl), 1.0);
    let dhdz = (terrain[idx(x, zd)] - terrain[idx(x, zu)]) / max(f32(zd - zu), 1.0);
    let grad = sqrt(dhdx * dhdx + dhdz * dhdz);
    let sin_tilt = grad / sqrt(1.0 + grad * grad);
    let tilt = max(sin_tilt, p.min_tilt);

    // Gate capacity by water depth so dry cells carry (and strand) no load,
    // and cap it: a cell may never suspend more than 5% of total relief,
    // whatever the (clamped) velocity says.
    let water_factor = clamp(water[i] / 0.01, 0.0, 1.0);
    let c = p.capacity_k * tilt * length(velocity[i]) * water_factor;
    capacity[i] = min(c, 0.05);
}

// Pass 5: dissolve terrain into sediment where capacity exceeds load,
// deposit where load exceeds capacity. Own cell only.
@compute @workgroup_size(8, 8, 1)
fn erode_deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let i = idx(gid.x, gid.y);
    let c = capacity[i];
    let s = sediment[i];
    if (c > s) {
        // Per-iteration dig cap (divergence guard, see spec error handling):
        // total_dig_budget normalized relief is distributed across all
        // iterations, so single channels can't outrun the rest of the field
        // and carve runaway slot canyons.
        let amt = min(p.erosion_k * (c - s) * p.dt, p.dig_cap);
        terrain[i] = terrain[i] - amt;
        sediment[i] = s + amt;
    } else {
        let amt = min(p.deposition_k * (s - c) * p.dt, s);
        terrain[i] = terrain[i] + amt;
        sediment[i] = s - amt;
    }
}

// Sediment advection, flux form: suspended load travels with the water that
// carries it, using the fractions already encoded in the flux field
// (fraction leaving via direction dir = flux_dir*dt / water_depth; the
// K-clamp in compute_flux guarantees the fractions sum to <= 1).
// Donor decrement and receiver credit use identical expressions, so total
// sediment is conserved to f32 rounding -- unlike a semi-Lagrangian gather,
// which amplifies load in converging (channelized) flow.
// Runs BEFORE water_velocity so `water` still holds the depths the flux
// scaling was computed against.
fn outflow_fraction(j: u32) -> f32 {
    let f = flux[j];
    return (f.x + f.y + f.z + f.w) * p.dt / max(water[j], 1e-6);
}

@compute @workgroup_size(8, 8, 1)
fn advect_sediment(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let x = gid.x;
    let z = gid.y;
    let i = idx(x, z);

    var s = sediment[i] * (1.0 - min(outflow_fraction(i), 1.0));
    if (x > 0u) {
        let j = idx(x - 1u, z);
        s += sediment[j] * flux[j].y * p.dt / max(water[j], 1e-6);
    }
    if (x + 1u < p.width) {
        let j = idx(x + 1u, z);
        s += sediment[j] * flux[j].x * p.dt / max(water[j], 1e-6);
    }
    if (z > 0u) {
        let j = idx(x, z - 1u);
        s += sediment[j] * flux[j].w * p.dt / max(water[j], 1e-6);
    }
    if (z + 1u < p.height) {
        let j = idx(x, z + 1u);
        s += sediment[j] * flux[j].z * p.dt / max(water[j], 1e-6);
    }
    sediment_new[i] = s;
}

// Final pass: evaporate water and clamp heights (divergence guard: bad
// params must not produce NaN/Inf terrain).
@compute @workgroup_size(8, 8, 1)
fn finalize_iter(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let i = idx(gid.x, gid.y);
    water[i] = water[i] * (1.0 - p.evaporation * p.dt);
    terrain[i] = clamp(terrain[i], -0.25, 1.25);
}

// Run once after the final iteration: ground still-suspended sediment so the
// terrain buffer accounts for the eroded mass. Cap the per-cell dump: in flow
// stagnation cells suspended load can grow unbounded over the run, and
// grounding it whole leaves a vertical stalagmite tower in the heightmap.
// Capping discards a tiny fraction of mass but eliminates the artifact.
@compute @workgroup_size(8, 8, 1)
fn settle(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let i = idx(gid.x, gid.y);
    let dump = min(sediment[i], 0.02);
    terrain[i] = clamp(terrain[i] + dump, -0.25, 1.25);
    sediment[i] = 0.0;
}
