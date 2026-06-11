// Thermal erosion: angle-of-repose (talus) smoothing on the GPU.
// Each thread reads its own cell + 4 axis neighbours from `terrain_in` and
// writes one cell to `terrain_out`, so the pass is race-free and deterministic.
// CPU side ping-pongs the two storage buffers between iterations.
//
// Pair-wise symmetric exchange: when cells A and B differ by more than max_drop,
// A's contribution from B and B's contribution from A are exactly opposite
// signs of the same magnitude, so total mass is conserved to f32 rounding.

struct Params {
    width: u32,
    height: u32,
    max_drop: f32,
    damping: f32,   // 0..1 fraction of the excess moved per iteration
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       terrain_in:  array<f32>;
@group(0) @binding(2) var<storage, read_write> terrain_out: array<f32>;

fn idx(x: u32, z: u32) -> u32 {
    return z * p.width + x;
}

// Height change `me` should accept from looking at `neighbour`.
// neighbour > me + max_drop  ->  receive (positive return)
// neighbour < me - max_drop  ->  shed    (negative return)
// otherwise                  ->  nothing
fn contribution(me: f32, neighbour: f32) -> f32 {
    let diff = neighbour - me;
    if (diff >  p.max_drop) { return (diff - p.max_drop) * p.damping * 0.5; }
    if (diff < -p.max_drop) { return (diff + p.max_drop) * p.damping * 0.5; }
    return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn talus_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) { return; }
    let x = gid.x;
    let z = gid.y;
    let i = idx(x, z);
    let me = terrain_in[i];

    var delta = 0.0;
    if (x > 0u)            { delta += contribution(me, terrain_in[idx(x - 1u, z)]); }
    if (x + 1u < p.width)  { delta += contribution(me, terrain_in[idx(x + 1u, z)]); }
    if (z > 0u)            { delta += contribution(me, terrain_in[idx(x, z - 1u)]); }
    if (z + 1u < p.height) { delta += contribution(me, terrain_in[idx(x, z + 1u)]); }

    terrain_out[i] = me + delta;
}
