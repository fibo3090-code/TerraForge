# Phase 6: Biome Texturing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the Phase 4-era per-vertex slope/height colouring with a real per-fragment biome material — four CC0 PBR sets (grass, dirt, rock, snow) blended by elevation and slope, with triplanar projection on steep faces and a detail-normal contribution that survives close camera distances.

**Architecture:** Bevy 0.18 `ExtendedMaterial<StandardMaterial, BiomeExtension>` — we keep all of `StandardMaterial`'s PBR + shadow plumbing and only override the fragment shader to compute albedo + perturbed normal from the four texture sets. World-space cell coords give us position-stable UVs (no UV seams from re-meshing). Rock + snow use triplanar projection (XY/YZ/XZ blended by the squared surface-normal components) so vertical cliff faces don't stretch. Grass + dirt use a single planar XZ projection — they only appear where the surface is nearly horizontal, so triplanar is wasted work there.

**Tech Stack:** existing `bevy = "0.18"` + `image = "0.25"` (already in tree). No new Rust crates. Shader at `assets/shaders/biome.wgsl` so Bevy's `AssetServer` watches it for hot-reload.

**Source spec:** build-order step 6 (`biome texturing — rock/grass/snow/sand by slope + altitude`); error handling (asset loads must surface, not silently fall back to a magenta material).

---

## File Structure

- `assets/biomes/{grass,dirt,rock,snow}_{albedo,normal}.jpg` — 1K JPG, CC0 from ambientCG (already in tree; `assets/biomes/README.md` attributes them).
- `assets/shaders/biome.wgsl` — fragment override.
- `src/biome.rs` — `BiomeExtension` (the `MaterialExtension` impl), `BiomeParams` uniform, helper that loads the eight textures via `AssetServer` and returns an `ExtendedMaterial<StandardMaterial, BiomeExtension>`.
- `src/main.rs` — add `MaterialPlugin::<ExtendedMaterial<StandardMaterial, BiomeExtension>>::default()`; replace the plain `StandardMaterial` spawn with the new material.
- `src/mesh_builder.rs` — drop `slope_height_colors` + the colour attribute; the mesh becomes pure geometry (positions, normals, UVs). Tests for `flat_heightmap_normals_point_up`, `grid_is_centred`, `height_maps_to_y_with_scale`, etc. stay green.

Material uniform (16-byte aligned, std140):

```rust
#[repr(C)]
struct BiomeParams {
    world_y_min:        f32,
    world_y_range:      f32,
    snow_line:          f32,  // normalized 0..1 height (post world_y_min/range)
    snow_blend:         f32,  // smoothstep width
    rock_slope_cos:     f32,  // cos(rock_slope_deg); steep iff normal.y < this
    rock_blend:         f32,  // smoothstep width on slope axis
    grass_dirt_line:    f32,  // normalized height (low) where grass -> dirt
    grass_dirt_blend:   f32,
    uv_scale_planar:    f32,  // 1 / world units per UV repeat for grass/dirt
    uv_scale_triplanar: f32,  // same, for rock/snow
    normal_strength:    f32,  // 0..1 mix of detail normal into surface normal
    _pad0:              f32,
}
```

Defaults: `snow_line = 0.78`, `snow_blend = 0.10`, `rock_slope_cos = cos(32°)`, `rock_blend = 0.15`, `grass_dirt_line = 0.30`, `grass_dirt_blend = 0.15`, `uv_scale_planar = 1/8` (one repeat per 8 world units), `uv_scale_triplanar = 1/6`, `normal_strength = 0.7`.

---

## Task 1: Drop vertex colours from `mesh_builder.rs`

**Files:** `src/mesh_builder.rs`

- [ ] **Step 1:** Remove `colors`, `slope_height_colors`, `smoothstep`, and the `ATTRIBUTE_COLOR` insertion. The mesh keeps positions / normals / UVs only.
- [ ] **Step 2:** `cargo nextest run mesh_builder` — the four existing tests (`vertex_and_index_counts_are_correct`, `flat_heightmap_normals_point_up`, `grid_is_centred_and_spans_world_size`, `height_maps_to_y_with_scale`) must stay green.

## Task 2: WGSL biome shader

**Files:** Create `assets/shaders/biome.wgsl`

- [ ] **Step 1:** Write the fragment override. Reuse Bevy's `pbr_fragment::pbr_input_from_standard_material` for normal/MR/AO and override **only** `base_color` and `N` (perturbed normal). Sketch:

```wgsl
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions as fns,
    forward_io::{VertexOutput, FragmentOutput},
    mesh_view_bindings::view,
}

struct BiomeParams { /* …16 floats above… */ };
@group(2) @binding(100) var<uniform> bp: BiomeParams;
@group(2) @binding(101) var grass_albedo: texture_2d<f32>;
@group(2) @binding(102) var grass_normal: texture_2d<f32>;
@group(2) @binding(103) var dirt_albedo:  texture_2d<f32>;
@group(2) @binding(104) var dirt_normal:  texture_2d<f32>;
@group(2) @binding(105) var rock_albedo:  texture_2d<f32>;
@group(2) @binding(106) var rock_normal:  texture_2d<f32>;
@group(2) @binding(107) var snow_albedo:  texture_2d<f32>;
@group(2) @binding(108) var snow_normal:  texture_2d<f32>;
@group(2) @binding(109) var biome_sampler: sampler;
```

Triplanar helper (returns weights for `xy`, `yz`, `xz` planes):

```wgsl
fn triplanar_weights(world_normal: vec3<f32>) -> vec3<f32> {
    let n = abs(world_normal);
    let w = pow(n, vec3<f32>(4.0));   // sharp blend
    return w / max(w.x + w.y + w.z, 1e-4);
}
```

`sample_triplanar(tex, world_pos, weights, scale)`: three `textureSample`s with UVs `pos.zy`, `pos.xz`, `pos.xy` scaled by `scale`; weighted sum.

Fragment:
1. `let h_norm = (world_pos.y - bp.world_y_min) / max(bp.world_y_range, 1e-4);` clamped to 0..1.
2. `let slope_t = 1.0 - smoothstep(bp.rock_slope_cos - bp.rock_blend, bp.rock_slope_cos + bp.rock_blend, in.world_normal.y);` — 0 on flat ground, 1 on steep faces.
3. Planar UV = `in.world_position.xz * bp.uv_scale_planar`; sample grass + dirt albedo/normal. Blend grass→dirt by `smoothstep(bp.grass_dirt_line - bp.grass_dirt_blend, …, h_norm)`.
4. Triplanar weights from `in.world_normal`; sample rock + snow albedo/normal at `uv_scale_triplanar`. Blend rock→snow by `smoothstep(bp.snow_line - …, …, h_norm)` *and* gate the snow contribution by `(1 - slope_t)` (snow only on gentle high ground).
5. Final albedo = `mix(low_blend, high_blend, slope_t)`. Final tangent-space normal = same mix on the *unpacked* normals (each sampled normal: `tex.xyz * 2 - 1`).
6. Build `pbr_input` from the standard material (so shadows / clearcoat / etc. keep working), then assign `pbr_input.material.base_color = vec4(final_albedo, 1.0);` and `pbr_input.N = perturbed_world_normal(pbr_input.N, sampled_tangent_normal, bp.normal_strength)` — a one-line slerp / mix-then-renormalise is sufficient since the surface is not heavily tangent-mapped.
7. Return `fns::apply_pbr_lighting(pbr_input)`.

- [ ] **Step 2:** Make sure UV scaling lines up with world units, not mesh space — `in.world_position` is the post-transform vertex position, so multiplying by `uv_scale_planar` directly gives "repeats per world unit". At `WORLD_SIZE = 100` and `uv_scale = 1/8` we get ~12 grass repeats across the terrain — good detail without obvious tiling.

## Task 3: `BiomeExtension` Rust binding

**Files:** Create `src/biome.rs`; modify `src/main.rs` (mod + plugin + spawn).

- [ ] **Step 1:** Define the extension:

```rust
use bevy::{
    asset::{Asset, Handle},
    pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial},
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
};

pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, BiomeExtension>;

#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct BiomeExtension {
    #[uniform(100)]
    pub params: BiomeParams,
    #[texture(101)] pub grass_albedo: Handle<Image>,
    #[texture(102)] pub grass_normal: Handle<Image>,
    #[texture(103)] pub dirt_albedo:  Handle<Image>,
    #[texture(104)] pub dirt_normal:  Handle<Image>,
    #[texture(105)] pub rock_albedo:  Handle<Image>,
    #[texture(106)] pub rock_normal:  Handle<Image>,
    #[texture(107)] pub snow_albedo:  Handle<Image>,
    #[texture(108)] pub snow_normal:  Handle<Image>,
    #[sampler(109)] pub biome_sampler: Handle<Image>,  // reuses any albedo for the sampler binding
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
pub struct BiomeParams { /* fields as above */ }

impl MaterialExtension for BiomeExtension {
    fn fragment_shader() -> ShaderRef { "shaders/biome.wgsl".into() }
}
```

- [ ] **Step 2:** Helper `load_terrain_material(asset_server, materials, y_min, y_range) -> Handle<TerrainMaterial>` that loads the 8 JPGs from `biomes/…` and constructs the asset.

- [ ] **Step 3:** Image color-space note — Bevy treats untyped JPG/PNG asset loads as sRGB by default. Albedos want sRGB, normals want linear. After loading, fetch the normal `Image` and call `image.texture_descriptor.format = TextureFormat::Rgba8Unorm`, OR load normals via `AssetServer::load_with_settings::<Image>("…", |s: &mut ImageLoaderSettings| { s.is_srgb = false; })`. Use the load-settings approach — it's local to the call site.

## Task 4: Wire into the app

**Files:** `src/main.rs`

- [ ] **Step 1:** Add `mod biome;` and `use biome::{BiomeParams, TerrainMaterial};`. Register the plugin in `main()`:

```rust
.add_plugins(MaterialPlugin::<TerrainMaterial>::default())
```

- [ ] **Step 2:** In `setup`, compute `y_min` and `y_range` from the post-thermal `hm` (already available in the logged stats line). Replace the plain `StandardMaterial` spawn with `MeshMaterial3d(materials.add(load_terrain_material(...)))`. Drop the per-vertex colour path entirely — the mesh now ships positions/normals/UVs only and the BiomeExtension does the rest.

- [ ] **Step 3:** `cargo build --release` — clean. **Step 4:** `cargo run --release` — terrain reads as grass on the low/flat ground, dirt on the mid-elevation rolling hills, rock on every steep face, and snow on the high gentle ground (peaks). Cliff faces show clear rock detail; grass / dirt have visible PBR roughness.

- [ ] **Step 5:** Visual checkpoint — capture a screenshot and tune `snow_line` / `rock_slope_cos` if biomes appear at the wrong elevation or slope. The thresholds are uniform, so tuning is a one-line constant change.

## Task 5: Verification

**Files:** `src/analysis.rs` (extend the smoke test).

- [ ] **Step 1:** Add a follow-up `pipeline_smoke` assertion (not a new test): after computing `compute_stats(&thermal)`, derive what the biome thresholds *should* land at given the stats (e.g. snow only above `mean + 1 σ`, grass only below `mean − 1 σ`). Print them — this is the first sanity gate when biomes look wrong.
- [ ] **Step 2:** `cargo nextest run` — full suite (32 tests + 1 smoke), all green.

---

## Self-Review

**Spec coverage:** Biome texturing by slope + altitude ✓ (per-fragment, four bands); CC0 PBR set on disk ✓; triplanar on steep faces ✓; detail normals contribute to lighting ✓; Bevy's PBR shadow/lighting plumbing reused (no re-implementation) ✓. **Out of scope:** water plane + atmospheric sky (Phase 7); egui live controls (Phase 8).

**Type consistency:** `TerrainMaterial = ExtendedMaterial<StandardMaterial, BiomeExtension>` used identically in plugin registration and the spawn call.

**Asset hygiene:** all four texture sets sourced under CC0 from ambientCG; attribution in `assets/biomes/README.md`. Repo grows by ~15 MB total — within reason for a Rust gamedev prototype, no Git LFS needed.

## Definition of done
- App spawns terrain with `ExtendedMaterial<StandardMaterial, BiomeExtension>`.
- Visual check: grass on flat low ground, dirt mid-slope, rock on every steep face, snow on flat highs; detail normals visible at close camera distances.
- `cargo nextest run` 33-tests-green (32 unit/GPU + 1 smoke).
- No regressions in `mesh_builder` tests after dropping vertex colours.
- `target/` does not contain leaked assets; `assets/biomes/_dl/` does not exist (or is in `.gitignore`).
