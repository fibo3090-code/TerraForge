# Phase 2: Procedural Base (fBm noise) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the placeholder cosine heightmap with believable large-scale terrain from multi-octave fractal Brownian motion (fBm) Perlin noise.

**Architecture:** A pure, testable `generate_fbm(width, height, &FbmParams) -> Heightmap` built on the `noise` crate's `Fbm<Perlin>`. `FbmParams` holds the tunables (seed, octaves, frequency, lacunarity, persistence, amplitude). `main.rs` calls it instead of the cosine function. No Bevy/GPU involvement — this slots into the existing `Heightmap → mesh` seam unchanged.

**Tech Stack:** `noise = "0.9"` (Perlin + Fbm/MultiFractal).

**Source spec:** build-order step 2 ("multi-octave fBm → believable large-scale shape").

**Builds on Phase 1:** reuses `Heightmap` and `heightmap_to_mesh` untouched.

---

## File Structure

- `Cargo.toml` — add `noise = "0.9"`.
- `src/terrain_noise.rs` — `FbmParams` + `generate_fbm`. Pure logic.
- `src/main.rs` — swap the cosine closure for `generate_fbm`.

---

## Task 1: Add the noise dependency

**Files:** `Cargo.toml`

- [ ] **Step 1: Add the crate**

Run: `cargo add noise@0.9`
Expected: adds `noise = "0.9"` under `[dependencies]`.

- [ ] **Step 2: Verify it resolves and builds**

Run: `cargo build`
Expected: compiles (noise is a small, fast crate).

---

## Task 2: `generate_fbm` (TDD)

**Files:**
- Create: `src/terrain_noise.rs`
- Modify: `src/main.rs` (add `mod terrain_noise;`)

- [ ] **Step 1: Write the module + failing tests**

Create `src/terrain_noise.rs`:

```rust
//! Procedural terrain base: multi-octave fBm Perlin noise -> Heightmap.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::heightmap::Heightmap;

/// Tunables for the fBm base layer. (Folded into the central TerrainParams
/// resource in a later phase; standalone and pure for now.)
#[derive(Clone, Debug)]
pub struct FbmParams {
    pub seed: u32,
    pub octaves: usize,
    /// Feature density across the [0,1] sampling domain. Higher = smaller features.
    pub frequency: f64,
    /// Per-octave frequency multiplier (typically ~2.0).
    pub lacunarity: f64,
    /// Per-octave amplitude multiplier (typically ~0.5).
    pub persistence: f64,
    /// World-unit height the normalized noise (~[-1,1]) is scaled to.
    pub amplitude: f32,
}

impl Default for FbmParams {
    fn default() -> Self {
        Self {
            seed: 0,
            octaves: 6,
            frequency: 2.0,
            lacunarity: 2.0,
            persistence: 0.5,
            amplitude: 20.0,
        }
    }
}

/// Sample multi-octave fBm Perlin noise over a `width`×`height` grid.
/// Deterministic in `params.seed`.
pub fn generate_fbm(width: usize, height: usize, params: &FbmParams) -> Heightmap {
    let fbm = Fbm::<Perlin>::new(params.seed)
        .set_octaves(params.octaves)
        .set_frequency(params.frequency)
        .set_lacunarity(params.lacunarity)
        .set_persistence(params.persistence);

    Heightmap::from_fn(width, height, |x, z| {
        let nx = x as f64 / (width - 1) as f64;
        let nz = z as f64 / (height - 1) as f64;
        (fbm.get([nx, nz]) as f32) * params.amplitude
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_for_same_seed() {
        let p = FbmParams::default();
        let a = generate_fbm(64, 64, &p);
        let b = generate_fbm(64, 64, &p);
        assert_eq!(a.data(), b.data());
    }

    #[test]
    fn different_seed_gives_different_terrain() {
        let a = generate_fbm(64, 64, &FbmParams { seed: 1, ..Default::default() });
        let b = generate_fbm(64, 64, &FbmParams { seed: 2, ..Default::default() });
        assert_ne!(a.data(), b.data());
    }

    #[test]
    fn output_is_not_constant() {
        let hm = generate_fbm(64, 64, &FbmParams::default());
        let first = hm.get(0, 0);
        assert!(hm.data().iter().any(|&h| (h - first).abs() > 1e-3),
            "fBm output should vary across the grid");
    }

    #[test]
    fn heights_stay_within_amplitude_bounds() {
        let p = FbmParams { amplitude: 20.0, ..Default::default() };
        let hm = generate_fbm(96, 96, &p);
        // fBm normalized output is within ~[-1,1]; allow a small margin.
        let bound = p.amplitude * 1.2;
        assert!(hm.data().iter().all(|&h| h.abs() <= bound),
            "height exceeded {bound}");
    }
}
```

- [ ] **Step 2: Register the module**

In `src/main.rs`, add `mod terrain_noise;` next to the other `mod` lines.

- [ ] **Step 3: Run the tests**

Run: `cargo test terrain_noise`
Expected: PASS (4 tests). If `Fbm`/`MultiFractal`/`set_octaves` fail to resolve, check the installed noise 0.9 API with `cargo doc -p noise --no-deps`; the builder methods live on the `MultiFractal` trait and `NoiseFn::get` on `NoiseFn`.

- [ ] **Step 4: Full suite**

Run: `cargo test`
Expected: Phase 1 (7) + Phase 2 (4) tests all pass.

---

## Task 3: Use fBm in the app (visual)

**Files:** `src/main.rs`

- [ ] **Step 1: Swap the placeholder for real noise**

In `src/main.rs`: add the import `use terrain_noise::{generate_fbm, FbmParams};`, and replace the `Heightmap::from_fn(GRID, GRID, |x, z| { ...cosine... })` block in `setup` with:

```rust
    let hm = generate_fbm(
        GRID,
        GRID,
        &FbmParams { amplitude: 18.0, ..Default::default() },
    );
```

Remove the now-unused `use heightmap::Heightmap;` if the compiler warns it is unused (the closure was its only user). Keep `mesh_builder::heightmap_to_mesh`.

- [ ] **Step 2: Build**

Run: `cargo build`
Expected: clean (no unused-import warnings).

- [ ] **Step 3: Run and verify**

Run: `cargo run`
Expected: window shows irregular, natural-looking rolling terrain (not the regular cosine waves). Orbit to inspect ridges/valleys. Closing the window exits.

- [ ] **Step 4: Commit (await user's go-ahead per workflow)**

Atomic commits: (a) `Cargo.toml`/`Cargo.lock` noise dep, (b) `src/terrain_noise.rs`, (c) `src/main.rs` swap, (d) the plan doc.

---

## Self-Review

**Spec coverage (step 2 — multi-octave fBm believable base):** `generate_fbm` with octaves/frequency/lacunarity/persistence → ✓. Determinism (a success criterion) explicitly tested → ✓. Tunables captured in `FbmParams`, the seed of the future `TerrainParams` → ✓.

**Placeholder scan:** none — all code complete.

**Type consistency:** `FbmParams` fields and `generate_fbm(usize, usize, &FbmParams) -> Heightmap` used identically in tests and `main.rs`. Reuses `Heightmap::{from_fn, get, data}` from Phase 1 unchanged.

## Definition of done
- `cargo test` green (11 tests).
- App renders natural fBm terrain.
- Deterministic for a given seed.
