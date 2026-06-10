# Phase 3: Tectonic Control — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give the fBm base large-scale geological structure — continental uplift zones and ridge-guided mountain ranges — instead of uniformly bumpy noise.

**Architecture:** A pure `apply_tectonics(&Heightmap, &TectonicParams) -> Heightmap` stage honouring the pipeline contract (heightmap in → heightmap out). Composition: `out = base + uplift_strength·U + ridge_strength·R·U`, where `U` is low-frequency fBm remapped to [0,1] (uplift zones) and `R` is ridged multifractal remapped to [0,1]. Multiplying `R` by `U` confines mountain ridges to uplift zones — geologically plausible structure. All terms are non-negative, so uplift monotonicity (`out ≥ base`, a spec test requirement) holds exactly.

**Tech Stack:** `noise = "0.9"` (`Fbm<Perlin>`, `RidgedMulti<Perlin>`, `MultiFractal`) — already a dependency.

**Source spec:** build-order step 3 ("uplift / ridge guidance for structured mountains"); testing section requires "tectonic uplift monotonicity".

**Builds on Phases 1–2:** consumes the `Heightmap` from `generate_fbm`; `heightmap_to_mesh` unchanged.

---

## File Structure

- `src/tectonics.rs` — `TectonicParams` + `apply_tectonics`. Pure logic, no Bevy.
- `src/main.rs` — insert tectonics stage between noise and mesh; raise GRID for ridge detail.

---

## Task 1: `apply_tectonics` (TDD)

**Files:**
- Create: `src/tectonics.rs`
- Modify: `src/main.rs` (add `mod tectonics;`)

- [ ] **Step 1: Write the module + tests**

Create `src/tectonics.rs`:

```rust
//! Tectonic control: large-scale uplift zones and ridge guidance applied to a
//! heightmap. out = base + uplift_strength*U + ridge_strength*R*U, with U a
//! low-frequency uplift field in [0,1] and R a ridged multifractal in [0,1].
//! Ridges are modulated by uplift so mountain ranges form only in uplift zones.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

use crate::heightmap::Heightmap;

/// Tunables for the tectonic stage.
#[derive(Clone, Debug)]
pub struct TectonicParams {
    pub seed: u32,
    /// Feature density of the uplift field over the [0,1] domain (low = continental scale).
    pub uplift_frequency: f64,
    /// World-unit height added where uplift is maximal.
    pub uplift_strength: f32,
    /// Feature density of the ridge field.
    pub ridge_frequency: f64,
    /// World-unit height of ridge crests in fully-uplifted zones.
    pub ridge_strength: f32,
}

impl Default for TectonicParams {
    fn default() -> Self {
        Self {
            seed: 0,
            uplift_frequency: 1.2,
            uplift_strength: 14.0,
            ridge_frequency: 3.0,
            ridge_strength: 16.0,
        }
    }
}

/// Remap noise output from ~[-1,1] to [0,1], clamped.
#[inline]
fn remap01(n: f64) -> f32 {
    ((n as f32) * 0.5 + 0.5).clamp(0.0, 1.0)
}

/// Apply uplift + ridge guidance to `base`. Pure; deterministic in `params.seed`.
/// Output dimensions equal input dimensions; output >= input everywhere
/// (monotone uplift) for non-negative strengths.
pub fn apply_tectonics(base: &Heightmap, params: &TectonicParams) -> Heightmap {
    let uplift = Fbm::<Perlin>::new(params.seed)
        .set_octaves(2)
        .set_frequency(params.uplift_frequency);
    let ridges = RidgedMulti::<Perlin>::new(params.seed.wrapping_add(1))
        .set_octaves(4)
        .set_frequency(params.ridge_frequency);

    let (w, h) = (base.width, base.height);
    Heightmap::from_fn(w, h, |x, z| {
        let nx = x as f64 / (w - 1) as f64;
        let nz = z as f64 / (h - 1) as f64;
        let u = remap01(uplift.get([nx, nz]));
        let r = remap01(ridges.get([nx, nz]));
        base.get(x, z) + params.uplift_strength * u + params.ridge_strength * r * u
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain_noise::{generate_fbm, FbmParams};

    fn base_64() -> Heightmap {
        generate_fbm(64, 64, &FbmParams::default())
    }

    #[test]
    fn preserves_dimensions() {
        let out = apply_tectonics(&base_64(), &TectonicParams::default());
        assert_eq!(out.width, 64);
        assert_eq!(out.height, 64);
    }

    #[test]
    fn is_deterministic_for_same_params() {
        let base = base_64();
        let p = TectonicParams::default();
        assert_eq!(apply_tectonics(&base, &p).data(), apply_tectonics(&base, &p).data());
    }

    #[test]
    fn zero_strengths_is_identity() {
        let base = base_64();
        let p = TectonicParams { uplift_strength: 0.0, ridge_strength: 0.0, ..Default::default() };
        assert_eq!(apply_tectonics(&base, &p).data(), base.data());
    }

    #[test]
    fn uplift_is_monotone_never_lowers_terrain() {
        let base = base_64();
        let out = apply_tectonics(&base, &TectonicParams::default());
        for z in 0..base.height {
            for x in 0..base.width {
                assert!(
                    out.get(x, z) >= base.get(x, z),
                    "lowered at ({x},{z}): {} -> {}", base.get(x, z), out.get(x, z)
                );
            }
        }
    }

    #[test]
    fn uplift_raises_mean_height() {
        let base = base_64();
        let out = apply_tectonics(&base, &TectonicParams::default());
        let mean = |hm: &Heightmap| hm.data().iter().sum::<f32>() / hm.data().len() as f32;
        assert!(mean(&out) > mean(&base) + 1.0, "uplift should raise mean height materially");
    }
}
```

- [ ] **Step 2: Register the module**

In `src/main.rs`, add `mod tectonics;` next to the other `mod` lines.

- [ ] **Step 3: Run the tests**

Run: `cargo test tectonics`
Expected: PASS (5 tests). If `RidgedMulti` import fails, check `cargo doc -p noise --no-deps` — it lives at the crate root in noise 0.9.

- [ ] **Step 4: Full suite**

Run: `cargo test`
Expected: 16 tests pass (7 Phase 1 + 4 Phase 2 + 5 Phase 3).

---

## Task 2: Wire tectonics into the app (visual)

**Files:** `src/main.rs`

- [ ] **Step 1: Insert the stage**

In `src/main.rs`:
- Add import: `use tectonics::{apply_tectonics, TectonicParams};`
- Raise `GRID` from 128 to 256 (ridge detail needs it; 65k vertices is trivial for the mesh path).
- Replace the heightmap generation in `setup` with:

```rust
    // Pipeline: fBm base -> tectonic uplift + ridge guidance.
    let base = generate_fbm(
        GRID,
        GRID,
        &FbmParams { amplitude: 10.0, ..Default::default() },
    );
    let hm = apply_tectonics(&base, &TectonicParams::default());
```

(Base amplitude drops 18 → 10: the tectonic stage now contributes the large-scale relief; the base supplies medium-scale texture.)

- [ ] **Step 2: Build**

Run: `cargo build`
Expected: clean, no warnings.

- [ ] **Step 3: Run and verify**

Run: `cargo run`
Expected: terrain now has *structure* — distinct mountainous regions with ridge lines, and lower rolling areas between them, instead of uniform bumps everywhere.

- [ ] **Step 4: Commit (await user's go-ahead per workflow)**

Atomic commits: (a) `src/tectonics.rs` + mod line, (b) `src/main.rs` pipeline wiring, (c) plan doc.

---

## Self-Review

**Spec coverage (step 3 — uplift / ridge guidance for structured mountains):** uplift field ✓ (low-freq fBm, strength-scaled); ridge guidance ✓ (RidgedMulti modulated by uplift, so ranges are spatially structured); spec's "tectonic uplift monotonicity" test ✓ (`uplift_is_monotone_never_lowers_terrain`). Slope constraints from the spec's pipeline sketch belong to thermal erosion (Phase 5, angle-of-repose) — not duplicated here.

**Placeholder scan:** none — all code complete.

**Type consistency:** `apply_tectonics(&Heightmap, &TectonicParams) -> Heightmap` used identically in tests and `main.rs`; reuses `Heightmap::{from_fn,get,data}` and `generate_fbm`/`FbmParams` exactly as defined in Phases 1–2.

## Definition of done
- `cargo test` green (16 tests).
- App shows structured mountain ranges with ridges in uplift zones.
- Deterministic per seed.
