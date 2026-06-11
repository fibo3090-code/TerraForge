// Biome fragment shader. Sits on top of Bevy 0.18's StandardMaterial via
// ExtendedMaterial: we let the base material handle MR/AO/emissive/shadow
// plumbing and override only `base_color` and the perturbed surface normal.
//
// Design:
//   - All four biomes (grass, dirt, rock, snow) sample with triplanar
//     projection. Planar XZ stretches badly on slopes; triplanar follows the
//     surface so cliffs and rolling hills both read correctly.
//   - A hash-based 2D value noise modulates (a) the UV scale per sample and
//     (b) the grass<->dirt boundary, so biomes break up into patches instead
//     of clean elevation bands and the obvious texture tile is hidden.
//   - Elevation picks grass vs dirt; slope picks low vs rock; snow is a
//     top-level overlay on high gentle ground.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{VertexOutput, FragmentOutput},
}

struct BiomeParams {
    world_y_min: f32,
    world_y_range: f32,
    snow_line: f32,
    snow_blend: f32,
    rock_slope_cos: f32,
    rock_blend: f32,
    grass_dirt_line: f32,
    grass_dirt_blend: f32,
    uv_scale_planar: f32,
    uv_scale_triplanar: f32,
    normal_strength: f32,
    _pad0: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> bp: BiomeParams;

@group(#{MATERIAL_BIND_GROUP}) @binding(101) var grass_albedo: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var grass_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var dirt_albedo:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var dirt_normal:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var rock_albedo:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var rock_normal:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(107) var snow_albedo:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(108) var snow_normal:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(109) var biome_sampler: sampler;

// ---- Cheap 2D value noise --------------------------------------------------
fn hash21(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Two octaves, output in [-1, 1].
fn macro_noise(p: vec2<f32>) -> f32 {
    let n = value_noise(p) * 0.6 + value_noise(p * 2.13 + vec2<f32>(31.7, 19.3)) * 0.4;
    return n * 2.0 - 1.0;
}

// ---- Triplanar sampling ----------------------------------------------------
// Softer power than the textbook (4) — the textbook value gives a tight
// transition between which plane dominates, but tight transitions on a noisy
// terrain surface show up as visible block seams. Power of 2 still suppresses
// the off-axis planes meaningfully on a clear face but blends smoothly when
// two faces are comparable.
// Sharp triplanar (power 8). A slope with normal.y=0.94 (~20° tilt) puts
// >97% of the weight on the Y plane, so the X/Z planes can't bleed sub-pixel
// interference into the result. Soft mixing of three planes was producing
// fine streaks on slopes; keep the dominant plane dominant.
fn triplanar_weights(world_normal: vec3<f32>) -> vec3<f32> {
    let n = abs(world_normal);
    let w = n * n * n * n * n * n * n * n;
    return w / max(w.x + w.y + w.z, 1e-4);
}

// Three texture samples in world space, weighted blend. No UV jitter — a
// multiplicative noise on `world_pos * scale` produces sub-pixel UV jumps of
// many texture repeats and reads as moire streaks.
fn sample_triplanar(
    tex: texture_2d<f32>,
    world_pos: vec3<f32>,
    weights: vec3<f32>,
    scale: f32,
) -> vec4<f32> {
    let sx = textureSample(tex, biome_sampler, world_pos.zy * scale);
    let sy = textureSample(tex, biome_sampler, world_pos.xz * scale);
    let sz = textureSample(tex, biome_sampler, world_pos.xy * scale);
    return sx * weights.x + sy * weights.y + sz * weights.z;
}

fn unpack_normal(s: vec4<f32>) -> vec3<f32> {
    return s.xyz * 2.0 - 1.0;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    let world_pos = in.world_position.xyz;
    let world_normal = normalize(in.world_normal);

    // Normalized elevation 0..1 over the post-erosion height range.
    let h_norm = clamp(
        (world_pos.y - bp.world_y_min) / max(bp.world_y_range, 1e-4),
        0.0, 1.0,
    );

    // slope_t in [0,1]: 0 on flat ground, 1 on near-vertical cliffs.
    let slope_t = 1.0 - smoothstep(
        bp.rock_slope_cos - bp.rock_blend,
        bp.rock_slope_cos + bp.rock_blend,
        world_normal.y,
    );

    // ----- Sample every biome via triplanar -------------------------------
    let tri_w = triplanar_weights(world_normal);
    let scale = bp.uv_scale_triplanar;
    let scale_low = bp.uv_scale_planar;  // grass/dirt repeat slower than rock
    let s_grass_a = sample_triplanar(grass_albedo, world_pos, tri_w, scale_low);
    let s_grass_n = unpack_normal(sample_triplanar(grass_normal, world_pos, tri_w, scale_low));
    let s_dirt_a  = sample_triplanar(dirt_albedo,  world_pos, tri_w, scale_low);
    let s_dirt_n  = unpack_normal(sample_triplanar(dirt_normal,  world_pos, tri_w, scale_low));
    let s_rock_a  = sample_triplanar(rock_albedo,  world_pos, tri_w, scale);
    let s_rock_n  = unpack_normal(sample_triplanar(rock_normal,  world_pos, tri_w, scale));
    let s_snow_a  = sample_triplanar(snow_albedo,  world_pos, tri_w, scale);
    let s_snow_n  = unpack_normal(sample_triplanar(snow_normal,  world_pos, tri_w, scale));

    // ----- Grass vs dirt with noise-modulated boundary --------------------
    // Without noise the boundary is a clean elevation band; with noise it
    // breaks into patches and reads as a natural meadow/dirt transition.
    // Clean elevation band — would like noise-modulated patchiness here, but
    // the `fract(sin) * 43758` hash in macro_noise has GPU precision artifacts
    // that bake directional streaks into the blend on slopes. TODO: replace
    // the hash with a PCG integer hash and re-enable.
    let gd_t = smoothstep(
        bp.grass_dirt_line - bp.grass_dirt_blend,
        bp.grass_dirt_line + bp.grass_dirt_blend,
        h_norm,
    );
    let low_albedo = mix(s_grass_a, s_dirt_a, gd_t);
    let low_normal = mix(s_grass_n, s_dirt_n, gd_t);

    // ----- Low <-> rock blend by slope ------------------------------------
    let base_albedo = mix(low_albedo, s_rock_a, slope_t);
    let base_normal = mix(low_normal, s_rock_n, slope_t);

    // ----- Snow overlay -----------------------------------------------------
    // Snow only on near-horizontal high ground. (Noise-modulated ragged edge
    // disabled until the macro_noise hash is replaced — see grass/dirt note.)
    let snow_t = smoothstep(
        bp.snow_line - bp.snow_blend,
        bp.snow_line + bp.snow_blend,
        h_norm,
    ) * (1.0 - smoothstep(0.1, 0.5, slope_t));
    let albedo = mix(base_albedo, s_snow_a, snow_t);
    let detail = mix(base_normal, s_snow_n, snow_t);

    // ----- Detail normal -> world space -----------------------------------
    // Tangent->world for planar XZ projection: (tx, ty, tz) -> world
    // (tx, tz, ty). Triplanar sampling makes the tangent frame plane-
    // dependent, but the approximation reads correctly on terrain and
    // normal_strength is gentle enough that residual error is invisible.
    let detail_world = normalize(vec3<f32>(detail.x, detail.z, detail.y));
    let perturbed = normalize(mix(world_normal, detail_world, bp.normal_strength));

    // ----- PBR plumbing ---------------------------------------------------
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = vec4<f32>(albedo.rgb, 1.0);
    pbr_input.N = perturbed;
    pbr_input.world_normal = perturbed;

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
