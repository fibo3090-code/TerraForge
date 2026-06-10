# Phase 1: Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get pixels on screen — a Bevy window showing a 3D terrain mesh generated from a heightmap, lit, with a fly-around (orbit) camera.

**Architecture:** A pure-logic core (`Heightmap` CPU type + `heightmap_to_mesh` builder) that is unit-testable with zero Bevy/GPU dependencies, wired into a thin Bevy app (`main.rs`) for windowing, lighting, and camera. This establishes the `heightmap in → mesh out` seam that every later stage (noise, tectonics, erosion) plugs into. GPU-side of `Heightmap` (the `R32Float` texture) is deferred to Phase 4 (erosion); Phase 1 only needs the CPU mirror.

**Tech Stack:** Rust (edition 2021), Bevy 0.18.1, bevy_panorbit_camera 0.34.

**Source spec:** `docs/superpowers/specs/2026-06-10-terrain-generator-design.md` (build-order step 1).

**Verified dependency facts (2026-06-10):** `bevy = "0.18"` is latest stable (0.19 is rc); `bevy_panorbit_camera = "0.34"` is compatible with bevy 0.18. (`bevy_atmosphere 0.13` still targets bevy 0.16 — a Phase 7 concern, not used here.)

---

## File Structure

- `Cargo.toml` — package + dependencies + dev opt-level profile (Bevy compiles slowly without optimized deps).
- `src/main.rs` — Bevy app: plugin wiring, `setup` system (camera, light, terrain mesh). Thin; no terrain logic.
- `src/heightmap.rs` — `Heightmap`: CPU `Vec<f32>` grid, `width`/`height`, `new`/`from_fn`/`get`/`set`/`data`. Pure, no Bevy.
- `src/mesh_builder.rs` — `heightmap_to_mesh(&Heightmap, world_size, height_scale) -> Mesh` + private `compute_normals`. Depends on `bevy::mesh` types only.

> Note: spec lists `render/mesh.rs`. For Phase 1 we use a flat `src/mesh_builder.rs`; it moves under a `render/` module in a later phase when `material.rs`/`water.rs` join it. Keep the function signature stable so the move is mechanical.

---

## Task 1: Scaffold Cargo project + window opens

**Files:**
- Create: `Cargo.toml`
- Create: `src/main.rs` (placeholder)

- [ ] **Step 1: Initialize the cargo package in-place**

The repo root already exists (git + docs/). Initialize a binary crate in place:

Run: `cargo init --bin --name terraforge`
Expected: creates `Cargo.toml` and `src/main.rs`. If `src/main.rs` already exists it is left untouched.

- [ ] **Step 2: Write `Cargo.toml`**

Replace the generated `Cargo.toml` with:

```toml
[package]
name = "terraforge"
version = "0.1.0"
edition = "2021"

[dependencies]
bevy = "0.18"
bevy_panorbit_camera = "0.34"

# Bevy is far too slow in a fully-unoptimized debug build.
# Optimize our code lightly and dependencies fully, while keeping fast incremental rebuilds.
[profile.dev]
opt-level = 1

[profile.dev.package."*"]
opt-level = 3
```

- [ ] **Step 3: Write a minimal `src/main.rs` that opens a window**

```rust
use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .run();
}
```

- [ ] **Step 4: Build (downloads + compiles Bevy — first build is slow, minutes)**

Run: `cargo build`
Expected: compiles with no errors. (First run downloads the Bevy tree; subsequent builds are incremental.)

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/main.rs .gitignore
git commit -m "feat: scaffold bevy terraforge crate, window opens"
```

---

## Task 2: `Heightmap` core type (TDD)

**Files:**
- Create: `src/heightmap.rs`
- Modify: `src/main.rs` (add `mod heightmap;`)

- [ ] **Step 1: Write the failing tests**

Create `src/heightmap.rs`:

```rust
//! CPU-side heightmap: an N×N grid of f32 heights.
//! Row-major: index = z * width + x. Every generation/erosion stage in the
//! pipeline consumes and produces this type (the GPU texture mirror is added
//! in the erosion phase).

/// A row-major grid of heights.
#[derive(Clone, Debug)]
pub struct Heightmap {
    pub width: usize,
    pub height: usize,
    data: Vec<f32>,
}

impl Heightmap {
    /// A `width`×`height` heightmap filled with zeros.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0.0; width * height] }
    }

    /// Build a heightmap by sampling `f(x, z)` over the grid.
    pub fn from_fn(width: usize, height: usize, f: impl Fn(usize, usize) -> f32) -> Self {
        let mut hm = Self::new(width, height);
        for z in 0..height {
            for x in 0..width {
                hm.set(x, z, f(x, z));
            }
        }
        hm
    }

    #[inline]
    pub fn get(&self, x: usize, z: usize) -> f32 {
        self.data[z * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, z: usize, value: f32) {
        self.data[z * self.width + x] = value;
    }

    /// Raw row-major heights, length `width * height`.
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_dims_and_is_zeroed() {
        let hm = Heightmap::new(4, 3);
        assert_eq!(hm.width, 4);
        assert_eq!(hm.height, 3);
        assert_eq!(hm.data().len(), 12);
        assert!(hm.data().iter().all(|&h| h == 0.0));
    }

    #[test]
    fn from_fn_indexes_row_major() {
        // value encodes (x, z) so we can verify indexing direction
        let hm = Heightmap::from_fn(3, 2, |x, z| (z * 3 + x) as f32);
        assert_eq!(hm.get(0, 0), 0.0);
        assert_eq!(hm.get(2, 0), 2.0);
        assert_eq!(hm.get(0, 1), 3.0);
        assert_eq!(hm.get(2, 1), 5.0);
    }

    #[test]
    fn set_then_get_roundtrips() {
        let mut hm = Heightmap::new(2, 2);
        hm.set(1, 0, 4.2);
        assert_eq!(hm.get(1, 0), 4.2);
        assert_eq!(hm.get(0, 0), 0.0);
    }
}
```

- [ ] **Step 2: Register the module so tests are discovered**

In `src/main.rs`, add the module declaration at the top (below the `use`):

```rust
use bevy::prelude::*;

mod heightmap;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .run();
}
```

- [ ] **Step 3: Run the tests**

Run: `cargo test heightmap`
Expected: PASS — `new_has_dims_and_is_zeroed`, `from_fn_indexes_row_major`, `set_then_get_roundtrips` (3 passed). The implementation is written alongside the tests in `heightmap.rs`, so they pass immediately; if any fails, fix `heightmap.rs` before continuing.

- [ ] **Step 4: Commit**

```bash
git add src/heightmap.rs src/main.rs
git commit -m "feat: add Heightmap CPU grid type with tests"
```

---

## Task 3: `heightmap_to_mesh` builder (TDD)

**Files:**
- Create: `src/mesh_builder.rs`
- Modify: `src/main.rs` (add `mod mesh_builder;`)

Geometry contract: the grid lies in the XZ plane centred on the origin, spanning `[-world_size/2, world_size/2]` in both X and Z; height maps to +Y scaled by `height_scale`. Vertex count = `width*height`. Triangle index count = `(width-1)*(height-1)*6`. Winding is chosen so a flat heightmap yields normals pointing +Y.

- [ ] **Step 1: Write the failing tests**

Create `src/mesh_builder.rs`:

```rust
//! Builds a renderable Bevy mesh from a `Heightmap`.
//! Grid is centred on the origin in the XZ plane; height is +Y.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::math::Vec3;

use crate::heightmap::Heightmap;

/// Convert a heightmap into a triangle-list mesh with positions, recomputed
/// normals, and UVs. `world_size` is the X/Z extent in world units; heights
/// are multiplied by `height_scale`.
pub fn heightmap_to_mesh(hm: &Heightmap, world_size: f32, height_scale: f32) -> Mesh {
    let (w, h) = (hm.width, hm.height);
    assert!(w >= 2 && h >= 2, "heightmap must be at least 2x2 to build a mesh");

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(w * h);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(w * h);

    for z in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32; // 0..=1
            let fz = z as f32 / (h - 1) as f32;
            let px = (fx - 0.5) * world_size;
            let pz = (fz - 0.5) * world_size;
            let py = hm.get(x, z) * height_scale;
            positions.push([px, py, pz]);
            uvs.push([fx, fz]);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity((w - 1) * (h - 1) * 6);
    for z in 0..h - 1 {
        for x in 0..w - 1 {
            let i = (z * w + x) as u32;
            let right = i + 1;
            let down = i + w as u32;
            let down_right = down + 1;
            // Winding chosen so flat ground normals point +Y (see compute_normals).
            indices.extend_from_slice(&[i, down, right]);
            indices.extend_from_slice(&[right, down, down_right]);
        }
    }

    let normals = compute_normals(&positions, &indices);

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

/// Area-weighted vertex normals from face normals.
fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let a = Vec3::from_array(positions[tri[0] as usize]);
        let b = Vec3::from_array(positions[tri[1] as usize]);
        let c = Vec3::from_array(positions[tri[2] as usize]);
        let face = (b - a).cross(c - a); // length ∝ triangle area
        for &idx in tri {
            normals[idx as usize] += face;
        }
    }
    normals
        .iter()
        .map(|n| n.normalize_or_zero().to_array())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    fn positions_of(mesh: &Mesh) -> Vec<[f32; 3]> {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("positions not Float32x3"),
        }
    }
    fn normals_of(mesh: &Mesh) -> Vec<[f32; 3]> {
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap() {
            VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("normals not Float32x3"),
        }
    }

    #[test]
    fn vertex_and_index_counts_are_correct() {
        let hm = Heightmap::new(4, 3); // flat
        let mesh = heightmap_to_mesh(&hm, 10.0, 1.0);
        assert_eq!(positions_of(&mesh).len(), 4 * 3);
        let n_indices = match mesh.indices().unwrap() {
            Indices::U32(v) => v.len(),
            Indices::U16(v) => v.len(),
        };
        assert_eq!(n_indices, (4 - 1) * (3 - 1) * 6);
    }

    #[test]
    fn flat_heightmap_normals_point_up() {
        let hm = Heightmap::new(5, 5); // all zero
        let mesh = heightmap_to_mesh(&hm, 8.0, 3.0);
        for n in normals_of(&mesh) {
            assert!((n[0]).abs() < 1e-5, "nx={} not ~0", n[0]);
            assert!((n[1] - 1.0).abs() < 1e-5, "ny={} not ~1 (winding wrong?)", n[1]);
            assert!((n[2]).abs() < 1e-5, "nz={} not ~0", n[2]);
        }
    }

    #[test]
    fn grid_is_centred_and_spans_world_size() {
        let hm = Heightmap::new(3, 3);
        let world = 6.0_f32;
        let mesh = heightmap_to_mesh(&hm, world, 1.0);
        let p = positions_of(&mesh);
        // first vertex (x=0,z=0) is the min corner, last is the max corner
        assert_eq!(p[0], [-world / 2.0, 0.0, -world / 2.0]);
        assert_eq!(*p.last().unwrap(), [world / 2.0, 0.0, world / 2.0]);
    }

    #[test]
    fn height_maps_to_y_with_scale() {
        let hm = Heightmap::from_fn(2, 2, |x, _z| if x == 1 { 2.0 } else { 0.0 });
        let mesh = heightmap_to_mesh(&hm, 4.0, 5.0);
        let p = positions_of(&mesh);
        // vertex at x=0 -> y 0; vertex at x=1 -> y = 2.0 * 5.0
        assert_eq!(p[0][1], 0.0);
        assert_eq!(p[1][1], 10.0);
    }
}
```

- [ ] **Step 2: Register the module**

In `src/main.rs` add `mod mesh_builder;` next to `mod heightmap;`:

```rust
use bevy::prelude::*;

mod heightmap;
mod mesh_builder;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .run();
}
```

- [ ] **Step 3: Run the tests**

Run: `cargo test mesh_builder`
Expected: PASS (4 tests). If `flat_heightmap_normals_point_up` fails with `ny=-1`, the triangle winding is reversed — swap to `[i, right, down]` / `[right, down_right, down]`. If imports fail to resolve (`bevy::mesh` / `bevy::asset::RenderAssetUsages`), confirm against the installed bevy 0.18 paths: try `bevy::render::mesh::{Indices, Mesh, PrimitiveTopology}` and `bevy::render::render_asset::RenderAssetUsages` as fallbacks, then re-run.

- [ ] **Step 4: Run the full test suite**

Run: `cargo test`
Expected: all heightmap + mesh_builder tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mesh_builder.rs src/main.rs
git commit -m "feat: build terrain mesh from heightmap with computed normals"
```

---

## Task 4: Bevy app — render terrain with fly camera + lighting (visual)

**Files:**
- Modify: `src/main.rs`

This task is visual/manual (no unit test — the window is the acceptance check, per the spec's testing section).

- [ ] **Step 1: Write the full app**

Replace `src/main.rs` with:

```rust
use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

mod heightmap;
mod mesh_builder;

use heightmap::Heightmap;
use mesh_builder::heightmap_to_mesh;

const GRID: usize = 128;
const WORLD_SIZE: f32 = 100.0;
const HEIGHT_SCALE: f32 = 1.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Trivial placeholder heightmap: two crossed cosine waves -> a gentle hill
    // field, just to prove generation -> mesh -> render. Real noise is Phase 2.
    let hm = Heightmap::from_fn(GRID, GRID, |x, z| {
        let fx = x as f32 / (GRID - 1) as f32;
        let fz = z as f32 / (GRID - 1) as f32;
        let h = (fx * std::f32::consts::TAU).cos() + (fz * std::f32::consts::TAU).cos();
        h * 6.0
    });
    let mesh = heightmap_to_mesh(&hm, WORLD_SIZE, HEIGHT_SCALE);

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.45, 0.42, 0.38),
            perceptual_roughness: 0.95,
            ..default()
        })),
    ));

    // Sun
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(50.0, 80.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Fly-around (orbit) camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 60.0, 110.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
```

- [ ] **Step 2: Build**

Run: `cargo build`
Expected: compiles cleanly. If `PanOrbitCamera`/`PanOrbitCameraPlugin` import paths differ, run `cargo doc -p bevy_panorbit_camera --no-deps` or check `cargo tree -i bevy_panorbit_camera`; the crate re-exports both from its root.

- [ ] **Step 3: Run and visually verify**

Run: `cargo run`
Expected: a window opens showing a lit, undulating terrain mesh. Left-drag orbits, right-drag pans, scroll zooms. Terrain is shaded (normals correct — not flat-black or inside-out).

- [ ] **Step 4: Capture an acceptance screenshot**

With the window focused, take a screenshot and save it to `docs/superpowers/specs/phase1-foundation.png` for the spec record (manual; OS screenshot tool). Confirm the terrain is visibly 3D and lit.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat: render terrain mesh with orbit camera and directional light"
```

---

## Self-Review

**Spec coverage (build-order step 1 — "Bevy app + window + fly camera + lighting; render a mesh from a trivial heightmap. Get pixels on screen."):**
- Bevy app + window → Task 1. ✓
- Render a mesh from a trivial heightmap → Tasks 2–4 (Heightmap → mesh → spawned). ✓
- Fly camera → Task 4 (PanOrbitCamera). ✓
- Lighting → Task 4 (DirectionalLight + shadows). ✓
- Establishes `heightmap in → mesh out` seam for later phases → Tasks 2–3. ✓

Deliberately deferred (later phases, per spec): real fBm noise (Phase 2), tectonics (3), GPU erosion (4–5), biome material (6), water/sky (7), egui sliders (8), `Heightmap` GPU texture mirror (Phase 4). `params.rs`/`TerrainParams` resource arrives when there are tunables to drive (Phase 2+); Phase 1 uses module consts.

**Placeholder scan:** No TBD/TODO/"handle errors" placeholders; every code step is complete and compilable.

**Type consistency:** `Heightmap::{new, from_fn, get, set, data}` and `heightmap_to_mesh(&Heightmap, f32, f32) -> Mesh` are used identically across Tasks 2–4. Module names (`heightmap`, `mesh_builder`) consistent in every `main.rs` revision.

## Definition of done
- `cargo test` green (Heightmap + mesh_builder).
- `cargo run` shows a lit, orbit-navigable 3D terrain mesh.
- Acceptance screenshot saved.
- Each task committed.
