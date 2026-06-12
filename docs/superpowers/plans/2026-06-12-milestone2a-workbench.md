# Milestone 2A: Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Incremental cached pipeline, async regeneration, map-size + full parameter exposure, multi-format heightmap export, and a one-command verification protocol — per the approved spec `docs/superpowers/specs/2026-06-12-workbench-design.md`.

**Architecture:** The pipeline moves out of `main.rs` into `src/pipeline.rs` as a four-stage cascade with per-stage param structs; each cache entry stores the exact param tuple that produced it, so dirtiness cascades structurally. Regeneration runs on Bevy's `AsyncComputeTaskPool` (the headless `GpuContext` is `Send + Sync` behind an `Arc`); a poll system swaps the mesh asset on completion. Export is a set of pure `(heightmap, …, path) -> Result` writers in `src/export.rs`.

**Tech Stack:** existing Bevy 0.18 + wgpu 27; new deps: `rfd` (native folder dialog), `exr` (float EXR write), `serde`/`serde_json` (metadata).

---

## File Structure

- Create: `src/pipeline.rs` — `BaseParams`, `TectonicSettings`, `HydraulicSettings`, `ThermalSettings`, `PipelineParams`, `StageCache`, `PipelineRun`, `run_pipeline` + tests.
- Create: `src/export.rs` — `write_png16`, `write_r16`, `write_exr`, `write_biome_mask`, `write_water_depth16`, `write_metadata`, `export_all` + tests.
- Create: `tools/verify.ps1` — test suite + dual-viewpoint capture.
- Modify: `src/main.rs` — replace `GenParams` with `PipelineParams`, world-size derivations, async regen, expanded panel, export section, post-regen audit.
- Modify: `Cargo.toml` — new deps.

---

### Task 1: Dependencies

- [ ] **Step 1:** Run:
```bash
cargo add rfd exr serde_json
cargo add serde --features derive
```
- [ ] **Step 2:** `cargo check` — clean (new deps unused yet; allow warnings only).
- [ ] **Step 3:** Commit: `chore: add rfd, exr, serde deps for workbench`

### Task 2: `src/pipeline.rs` — cached cascade (TDD)

**Files:** Create `src/pipeline.rs`; modify `src/main.rs` (add `mod pipeline;` only, keep old path compiling).

- [ ] **Step 1: Write the module skeleton with params + cache types** (full source):

```rust
//! Cached four-stage generation pipeline:
//! fBm base -> tectonics -> hydraulic erosion -> thermal erosion.
//! Each stage's cache entry stores the exact params that produced it, so a
//! param change re-runs only the affected suffix of the cascade.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::erosion::{self, GpuContext};
use crate::heightmap::Heightmap;
use crate::tectonics::{apply_tectonics, TectonicParams};
use crate::terrain_noise::{generate_fbm, FbmParams};

pub const TALUS_ITERATIONS: u32 = 12;
/// Render scale: 1 world unit = 20 m (drives the panel's km/m display only).
pub const METERS_PER_UNIT: f32 = 20.0;

#[derive(Clone, PartialEq, serde::Serialize)]
pub struct BaseParams {
    pub seed: u32,
    pub grid: usize,
    /// World extent in render units (50–500; 1–10 km at METERS_PER_UNIT).
    pub world_size: f32,
    pub amplitude: f32,
    pub octaves: usize,
    pub frequency: f32,
    pub persistence: f32,
}

#[derive(Clone, PartialEq, serde::Serialize)]
pub struct TectonicSettings {
    pub uplift_strength: f32,
    pub ridge_strength: f32,
    pub uplift_frequency: f32,
    pub ridge_frequency: f32,
}

#[derive(Clone, PartialEq, serde::Serialize)]
pub struct HydraulicSettings {
    pub rain_rate: f32,
    pub capacity_k: f32,
    pub total_dig_budget: f32,
}

#[derive(Clone, PartialEq, serde::Serialize)]
pub struct ThermalSettings {
    pub talus_angle_deg: f32,
}

#[derive(Clone, PartialEq, serde::Serialize)]
pub struct PipelineParams {
    pub base: BaseParams,
    pub tectonics: TectonicSettings,
    pub hydraulic: HydraulicSettings,
    pub thermal: ThermalSettings,
}

impl Default for PipelineParams {
    fn default() -> Self {
        Self {
            base: BaseParams {
                seed: 0,
                grid: 2048,
                world_size: 100.0,
                amplitude: 10.0,
                octaves: 6,
                frequency: 2.0,
                persistence: 0.5,
            },
            tectonics: TectonicSettings {
                uplift_strength: 14.0,
                ridge_strength: 16.0,
                uplift_frequency: 1.2,
                ridge_frequency: 3.0,
            },
            hydraulic: HydraulicSettings {
                rain_rate: 0.012,
                capacity_k: 0.8,
                total_dig_budget: 0.2,
            },
            thermal: ThermalSettings { talus_angle_deg: 35.0 },
        }
    }
}

/// Cached stage outputs, keyed by the exact param tuple that produced them.
/// Tuples include all upstream params, so dirtiness cascades structurally.
#[derive(Default)]
pub struct StageCache {
    base: Option<(BaseParams, Heightmap)>,
    tectonics: Option<(BaseParams, TectonicSettings, Heightmap)>,
    hydraulic: Option<(BaseParams, TectonicSettings, HydraulicSettings, Heightmap)>,
}

/// Progress stages written to the shared atomic during a run.
pub mod progress {
    pub const BASE: u8 = 0;
    pub const TECTONICS: u8 = 1;
    pub const HYDRAULIC: u8 = 2;
    pub const THERMAL: u8 = 3;
    pub const MESH: u8 = 4;
    pub const DONE: u8 = 5;
}

pub struct PipelineRun {
    pub heightmap: Heightmap,
    pub h_min: f32,
    pub h_max: f32,
    /// Relief of the (possibly cached) tectonic stage — audit baseline.
    pub tectonic_relief: f32,
    /// Stage label + seconds, only for stages that actually ran.
    pub stages_run: Vec<(&'static str, f32)>,
    pub reused: Vec<&'static str>,
}

/// Feature density per km stays constant across map sizes: noise frequencies
/// scale with world_size relative to the 100-unit baseline.
fn freq_scale(world_size: f32) -> f64 {
    (world_size / 100.0) as f64
}

pub fn run_pipeline(
    gpu: &GpuContext,
    p: &PipelineParams,
    cache: &mut StageCache,
    progress: &AtomicU8,
) -> PipelineRun {
    let mut stages_run = Vec::new();
    let mut reused = Vec::new();

    // --- Stage 1: fBm base -------------------------------------------------
    progress.store(progress::BASE, Ordering::Relaxed);
    let base_hm = match &cache.base {
        Some((cp, hm)) if *cp == p.base => {
            reused.push("fBm");
            hm.clone()
        }
        _ => {
            let t = std::time::Instant::now();
            let hm = generate_fbm(
                p.base.grid,
                p.base.grid,
                &FbmParams {
                    seed: p.base.seed,
                    octaves: p.base.octaves,
                    frequency: p.base.frequency as f64 * freq_scale(p.base.world_size),
                    persistence: p.base.persistence as f64,
                    amplitude: p.base.amplitude,
                    ..Default::default()
                },
            );
            stages_run.push(("fBm", t.elapsed().as_secs_f32()));
            cache.base = Some((p.base.clone(), hm.clone()));
            hm
        }
    };

    // --- Stage 2: tectonics ------------------------------------------------
    progress.store(progress::TECTONICS, Ordering::Relaxed);
    let tect_hm = match &cache.tectonics {
        Some((cb, ct, hm)) if *cb == p.base && *ct == p.tectonics => {
            reused.push("tectonics");
            hm.clone()
        }
        _ => {
            let t = std::time::Instant::now();
            let hm = apply_tectonics(
                &base_hm,
                &TectonicParams {
                    seed: p.base.seed,
                    uplift_strength: p.tectonics.uplift_strength,
                    ridge_strength: p.tectonics.ridge_strength,
                    uplift_frequency: p.tectonics.uplift_frequency as f64
                        * freq_scale(p.base.world_size),
                    ridge_frequency: p.tectonics.ridge_frequency as f64
                        * freq_scale(p.base.world_size),
                },
            );
            stages_run.push(("tectonics", t.elapsed().as_secs_f32()));
            cache.tectonics = Some((p.base.clone(), p.tectonics.clone(), hm.clone()));
            hm
        }
    };
    let (t_min, t_max) = min_max(&tect_hm);
    let tectonic_relief = t_max - t_min;

    // --- Stage 3: hydraulic erosion -----------------------------------------
    progress.store(progress::HYDRAULIC, Ordering::Relaxed);
    let hydro_hm = match &cache.hydraulic {
        Some((cb, ct, ch, hm))
            if *cb == p.base && *ct == p.tectonics && *ch == p.hydraulic =>
        {
            reused.push("hydraulic");
            hm.clone()
        }
        _ => {
            let t = std::time::Instant::now();
            let iterations = (p.base.grid as u32 * 3) / 4;
            let hm = erosion::erode_hydraulic(
                gpu,
                &tect_hm,
                &erosion::ErosionParams {
                    iterations,
                    rain_rate: p.hydraulic.rain_rate,
                    capacity_k: p.hydraulic.capacity_k,
                    total_dig_budget: p.hydraulic.total_dig_budget,
                    ..Default::default()
                },
            )
            .expect("hydraulic erosion failed");
            stages_run.push(("hydraulic", t.elapsed().as_secs_f32()));
            cache.hydraulic = Some((
                p.base.clone(),
                p.tectonics.clone(),
                p.hydraulic.clone(),
                hm.clone(),
            ));
            hm
        }
    };

    // --- Stage 4: thermal erosion (never cached: it's the live output) ------
    progress.store(progress::THERMAL, Ordering::Relaxed);
    let t = std::time::Instant::now();
    let cell_size = p.base.world_size / p.base.grid as f32;
    let max_drop = p.thermal.talus_angle_deg.to_radians().tan() * cell_size;
    let hm = erosion::erode_thermal(
        gpu,
        &hydro_hm,
        &erosion::ThermalParams {
            iterations: TALUS_ITERATIONS,
            max_drop,
            damping: 0.5,
        },
    )
    .expect("thermal erosion failed");
    stages_run.push(("thermal", t.elapsed().as_secs_f32()));

    let (h_min, h_max) = min_max(&hm);
    assert!(h_min.is_finite() && h_max.is_finite(), "pipeline produced NaN/Inf");

    PipelineRun { heightmap: hm, h_min, h_max, tectonic_relief, stages_run, reused }
}

fn min_max(hm: &Heightmap) -> (f32, f32) {
    hm.data()
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)))
}
```

Note: `ErosionParams` does not yet expose `rain_rate`/`capacity_k`/`total_dig_budget` overrides — it already has these fields, so struct-update syntax works as-is. `TectonicParams` already has the four fields used.

- [ ] **Step 2: Write the failing tests** (same file, `#[cfg(test)] mod tests`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> GpuContext {
        GpuContext::new().expect("pipeline tests require a GPU adapter")
    }

    fn small_params() -> PipelineParams {
        let mut p = PipelineParams::default();
        p.base.grid = 64;
        p
    }

    /// THE caching guard: for every stage boundary, an incremental run from
    /// a warm cache must be bit-identical to a cold full run.
    #[test]
    fn incremental_equals_full() {
        let gpu = ctx();
        let progress = AtomicU8::new(0);
        let mutations: Vec<(&str, Box<dyn Fn(&mut PipelineParams)>)> = vec![
            ("base", Box::new(|p: &mut PipelineParams| p.base.seed += 1)),
            ("tectonics", Box::new(|p: &mut PipelineParams| p.tectonics.uplift_strength += 2.0)),
            ("hydraulic", Box::new(|p: &mut PipelineParams| p.hydraulic.rain_rate *= 1.5)),
            ("thermal", Box::new(|p: &mut PipelineParams| p.thermal.talus_angle_deg += 3.0)),
        ];
        for (label, mutate) in mutations {
            let p1 = small_params();
            let mut cache = StageCache::default();
            let _ = run_pipeline(&gpu, &p1, &mut cache, &progress);
            let mut p2 = p1.clone();
            mutate(&mut p2);
            let incremental = run_pipeline(&gpu, &p2, &mut cache, &progress);
            let full = run_pipeline(&gpu, &p2, &mut StageCache::default(), &progress);
            assert_eq!(
                incremental.heightmap.data(),
                full.heightmap.data(),
                "incremental != full after mutating {label} params"
            );
        }
    }

    /// A thermal-only change must reuse all three cached upstream stages.
    #[test]
    fn thermal_change_reuses_upstream() {
        let gpu = ctx();
        let progress = AtomicU8::new(0);
        let p1 = small_params();
        let mut cache = StageCache::default();
        let first = run_pipeline(&gpu, &p1, &mut cache, &progress);
        assert_eq!(first.reused.len(), 0, "cold run must run everything");
        let mut p2 = p1.clone();
        p2.thermal.talus_angle_deg += 3.0;
        let second = run_pipeline(&gpu, &p2, &mut cache, &progress);
        assert_eq!(second.reused, vec!["fBm", "tectonics", "hydraulic"]);
        assert_eq!(second.stages_run.len(), 1);
        assert_eq!(second.stages_run[0].0, "thermal");
    }

    /// world_size participates in the base fingerprint (it scales noise
    /// frequency), so changing it must invalidate everything.
    #[test]
    fn world_size_invalidates_base() {
        let gpu = ctx();
        let progress = AtomicU8::new(0);
        let p1 = small_params();
        let mut cache = StageCache::default();
        let _ = run_pipeline(&gpu, &p1, &mut cache, &progress);
        let mut p2 = p1.clone();
        p2.base.world_size = 200.0;
        let run = run_pipeline(&gpu, &p2, &mut cache, &progress);
        assert!(run.reused.is_empty(), "world_size change must re-run all stages");
    }
}
```

- [ ] **Step 3:** Add `mod pipeline;` to `main.rs`. Run `cargo nextest run pipeline_` — expect FAIL/compile errors until Heightmap derives Clone (it already does) and `TectonicParams` field names line up. Fix signatures only, not test intent.
- [ ] **Step 4:** Run `cargo nextest run -E 'test(incremental) or test(thermal_change) or test(world_size_inv)'` — expect 3 PASS.
- [ ] **Step 5:** Full suite `cargo nextest run` — 36+ green. Commit: `feat: cached incremental pipeline with bit-identity guarantee`

### Task 3: Wire `main.rs` to the cached pipeline (still synchronous)

**Files:** Modify `src/main.rs`.

- [ ] **Step 1:** Delete `GenParams`; `PipelineParams` becomes the resource. Panel sliders map to `p.base.*`, `p.tectonics.*`, `p.hydraulic.*`, `p.thermal.*`. Add the new sliders (octaves 1–10, frequency 0.5–8.0, persistence 0.2–0.8, uplift_frequency 0.2–4.0, ridge_frequency 0.5–8.0, rain_rate 0.002–0.05, capacity_k 0.2–2.0, total_dig_budget 0.05–0.5, world_size 50–500 displayed as `{:.1} km` via `METERS_PER_UNIT`). Show derived cell size: `world_size / grid * METERS_PER_UNIT` metres. Add 🎲 button:
```rust
if ui.button("🎲 random seed").clicked() {
    p.base.seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() % 100_000;
    regen.0 = true;
}
```
- [ ] **Step 2:** `setup` and `regenerate_terrain` call `pipeline::run_pipeline(&gpu.0, &params, &mut cache, &progress)`; `StageCache` lives in a `ResMut<PipelineCache>` (`#[derive(Resource, Default)] struct PipelineCache(StageCache)`). Show reuse/timing summary in the panel from the last `PipelineRun` (`LastRunInfo` resource holding `String`).
- [ ] **Step 3:** World-size derivations: water plane size `world_size * 20.0`, shadow `maximum_distance: world_size * 4.0`, default camera `(0, 0.7 * world_size, 1.3 * world_size)`, capture cameras scale by `world_size / 100.0`. Water/camera/shadow update on regen if world_size changed (water plane: also replace its mesh via stored handle — store `water_mesh: Handle<Mesh>` in `TerrainHandles`).
- [ ] **Step 4:** `cargo build --release`; capture both viewpoints at defaults; compare against the Phase 8 pair (must match — defaults unchanged). Run `cargo nextest run` — green.
- [ ] **Step 5:** Commit: `feat: expanded settings + map size on cached pipeline`

### Task 4: Async regeneration

**Files:** Modify `src/main.rs`.

- [ ] **Step 1:** Resources + task plumbing:

```rust
use bevy::tasks::{AsyncComputeTaskPool, Task};
use bevy::tasks::futures_lite::future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Resource)]
struct GpuCompute(Arc<erosion::GpuContext>);

struct RegenOutput {
    run: pipeline::PipelineRun,
    mesh: Mesh,
    cache: pipeline::StageCache,
}

#[derive(Resource, Default)]
struct RegenInFlight {
    task: Option<Task<RegenOutput>>,
    progress: Arc<AtomicU8>,
}
```

`regenerate_terrain` becomes `spawn_regen`: on `RegenRequested`, if no task in flight, `take()` the cache out of `PipelineCache`, clone `PipelineParams` + `Arc<GpuContext>` + progress Arc, and:

```rust
let task = AsyncComputeTaskPool::get().spawn(async move {
    let run = pipeline::run_pipeline(&gpu, &params, &mut cache, &progress);
    progress.store(pipeline::progress::MESH, Ordering::Relaxed);
    let mesh = heightmap_to_mesh(&run.heightmap, params.base.world_size, HEIGHT_SCALE);
    progress.store(pipeline::progress::DONE, Ordering::Relaxed);
    RegenOutput { run, mesh, cache }
});
```

- [ ] **Step 2:** `poll_regen` system: `future::block_on(future::poll_once(&mut task))`; on `Some(output)`: mesh asset swap (`meshes.insert(&handles.mesh, output.mesh)`), stats/material/water updates as today, restore cache into `PipelineCache`, store `CurrentHeightmap(output.run.heightmap)` resource, build `LastRunInfo` string ("reused fBm, tectonics · hydraulic 2.4s · thermal 0.02s"), run the audit (Task 6).
- [ ] **Step 3:** Panel: while `RegenInFlight.task.is_some()`, replace the Regenerate button with a spinner + stage label from the progress atomic (`["fBm", "tectonics", "hydraulic", "thermal", "meshing", "done"][stage]`).
- [ ] **Step 4:** Manual check: run app, drag camera during a 2048² regen — no freeze; stage label advances. `cargo nextest run` green.
- [ ] **Step 5:** Commit: `feat: async regeneration with stage progress`

### Task 5: `src/export.rs` (TDD)

**Files:** Create `src/export.rs`; modify `src/main.rs` (mod + UI section).

- [ ] **Step 1: Write failing tests first** (in `src/export.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightmap::Heightmap;
    use std::fs;

    fn test_hm() -> Heightmap {
        Heightmap::from_fn(32, 32, |x, z| (x as f32 * 0.7) - (z as f32 * 0.3))
    }

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join("terraforge-export-tests").join(name);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn png16_roundtrip_within_quantization() {
        let hm = test_hm();
        let dir = tmp_dir("png16");
        let path = dir.join("h.png");
        let (mn, mx) = write_png16(&hm, &path).unwrap();
        let img = image::open(&path).unwrap().to_luma16();
        let range = mx - mn;
        for z in 0..hm.height {
            for x in 0..hm.width {
                let v = mn + (img.get_pixel(x as u32, z as u32).0[0] as f32 / 65535.0) * range;
                assert!((v - hm.get(x, z)).abs() <= range / 65535.0 + 1e-4);
            }
        }
    }

    #[test]
    fn exr_roundtrip_exact() {
        let hm = test_hm();
        let dir = tmp_dir("exr");
        let path = dir.join("h.exr");
        write_exr(&hm, &path).unwrap();
        let img = exr::prelude::read_first_rgba_layer_from_file(
            &path,
            |size, _| vec![0.0f32; size.width() * size.height()],
            |buf, pos, (r, _g, _b, _a): (f32, f32, f32, f32)| {
                buf[pos.y() * 32 + pos.x()] = r;
            },
        )
        .unwrap();
        for z in 0..32 {
            for x in 0..32 {
                assert_eq!(img.layer_data.channel_data.pixels[z * 32 + x], hm.get(x, z));
            }
        }
    }

    #[test]
    fn r16_length_and_endianness() {
        let hm = test_hm();
        let dir = tmp_dir("r16");
        let path = dir.join("h.r16");
        let (mn, mx) = write_r16(&hm, &path).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 32 * 32 * 2);
        // First sample: reconstruct and compare.
        let first = u16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 65535.0;
        let v = mn + first * (mx - mn);
        assert!((v - hm.get(0, 0)).abs() <= (mx - mn) / 65535.0 + 1e-4);
    }

    #[test]
    fn metadata_json_contains_scale_and_params() {
        let hm = test_hm();
        let dir = tmp_dir("meta");
        let path = dir.join("metadata.json");
        let params = crate::pipeline::PipelineParams::default();
        write_metadata(&hm, &params, 0.12, &path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(v["height_min_units"].is_number());
        assert!(v["world_size_meters"].is_number());
        assert_eq!(v["params"]["base"]["seed"], 0);
    }
}
```

- [ ] **Step 2:** Run `cargo nextest run export` — FAIL (functions missing).
- [ ] **Step 3: Implement** (full source above the tests):

```rust
//! Pure heightmap -> file writers. No Bevy types; everything is testable
//! headless. All errors are surfaced as Err(String) (spec: no silent export
//! failures).

use std::io::Write;
use std::path::Path;

use crate::analysis::{predict_biome, BiomeRules};
use crate::heightmap::Heightmap;
use crate::pipeline::{PipelineParams, METERS_PER_UNIT};

fn min_max(hm: &Heightmap) -> (f32, f32) {
    hm.data()
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)))
}

/// 16-bit grayscale PNG, min–max normalized. Returns the (min, max) used so
/// callers can record the scale.
pub fn write_png16(hm: &Heightmap, path: &Path) -> Result<(f32, f32), String> {
    let (mn, mx) = min_max(hm);
    let range = (mx - mn).max(1e-6);
    let img = image::ImageBuffer::<image::Luma<u16>, _>::from_fn(
        hm.width as u32,
        hm.height as u32,
        |x, z| {
            let v = (hm.get(x as usize, z as usize) - mn) / range;
            image::Luma([(v * 65535.0).round() as u16])
        },
    );
    img.save(path).map_err(|e| format!("png16 {}: {e}", path.display()))?;
    Ok((mn, mx))
}

/// Raw little-endian u16, same normalization as the PNG (Unreal RAW import).
pub fn write_r16(hm: &Heightmap, path: &Path) -> Result<(f32, f32), String> {
    let (mn, mx) = min_max(hm);
    let range = (mx - mn).max(1e-6);
    let mut bytes = Vec::with_capacity(hm.width * hm.height * 2);
    for z in 0..hm.height {
        for x in 0..hm.width {
            let v = ((hm.get(x, z) - mn) / range * 65535.0).round() as u16;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let mut f = std::fs::File::create(path).map_err(|e| format!("r16 {}: {e}", path.display()))?;
    f.write_all(&bytes).map_err(|e| format!("r16 {}: {e}", path.display()))?;
    Ok((mn, mx))
}

/// 32-bit float EXR with true world-unit heights in RGB (all channels equal —
/// universally readable; Blender/Gaea/Houdini take the R channel).
pub fn write_exr(hm: &Heightmap, path: &Path) -> Result<(), String> {
    let (w, h) = (hm.width, hm.height);
    exr::prelude::write_rgb_file(path, w, h, |x, y| {
        let v = hm.get(x, y);
        (v, v, v)
    })
    .map_err(|e| format!("exr {}: {e}", path.display()))
}

/// Predicted-biome colour map (same classifier as the analysis harness).
pub fn write_biome_mask(hm: &Heightmap, rules: &BiomeRules, path: &Path) -> Result<(), String> {
    crate::analysis::export_predicted_biome_png(hm, rules, path)
}

/// Water depth = max(0, sea_level - h), normalized over its own max.
pub fn write_water_depth16(hm: &Heightmap, sea_level: f32, path: &Path) -> Result<(), String> {
    let max_depth = hm
        .data()
        .iter()
        .map(|&v| (sea_level - v).max(0.0))
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let img = image::ImageBuffer::<image::Luma<u16>, _>::from_fn(
        hm.width as u32,
        hm.height as u32,
        |x, z| {
            let d = (sea_level - hm.get(x as usize, z as usize)).max(0.0) / max_depth;
            image::Luma([(d * 65535.0).round() as u16])
        },
    );
    img.save(path).map_err(|e| format!("water depth {}: {e}", path.display()))
}

/// Scale + reproducibility sidecar.
pub fn write_metadata(
    hm: &Heightmap,
    params: &PipelineParams,
    sea_level_frac: f32,
    path: &Path,
) -> Result<(), String> {
    let (mn, mx) = min_max(hm);
    let json = serde_json::json!({
        "height_min_units": mn,
        "height_max_units": mx,
        "height_min_meters": mn * METERS_PER_UNIT,
        "height_max_meters": mx * METERS_PER_UNIT,
        "world_size_units": params.base.world_size,
        "world_size_meters": params.base.world_size * METERS_PER_UNIT,
        "cell_size_meters": params.base.world_size / params.base.grid as f32 * METERS_PER_UNIT,
        "grid": params.base.grid,
        "sea_level_frac": sea_level_frac,
        "sea_level_units": mn + sea_level_frac * (mx - mn),
        "params": params,
    });
    std::fs::write(path, serde_json::to_string_pretty(&json).unwrap())
        .map_err(|e| format!("metadata {}: {e}", path.display()))
}

/// Write the full export set into `dir`. Returns the list of files written.
pub fn export_all(
    hm: &Heightmap,
    params: &PipelineParams,
    sea_level_frac: f32,
    dir: &Path,
) -> Result<Vec<String>, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let (mn, mx) = min_max(hm);
    let sea_level = mn + sea_level_frac * (mx - mn);
    let rules = BiomeRules::from_heightmap(hm, 32.0);
    let files = [
        ("height_16.png", write_png16(hm, &dir.join("height_16.png")).map(|_| ())),
        ("height.r16", write_r16(hm, &dir.join("height.r16")).map(|_| ())),
        ("height.exr", write_exr(hm, &dir.join("height.exr"))),
        ("biome_mask.png", write_biome_mask(hm, &rules, &dir.join("biome_mask.png"))),
        ("water_depth_16.png", write_water_depth16(hm, sea_level, &dir.join("water_depth_16.png"))),
        ("metadata.json", write_metadata(hm, params, sea_level_frac, &dir.join("metadata.json"))),
    ];
    let mut written = Vec::new();
    for (name, result) in files {
        result?;
        written.push(name.to_string());
    }
    Ok(written)
}
```

- [ ] **Step 4:** `cargo nextest run export` — 4 PASS. (`exr` read API names may need adjusting to the crate version — fix the test's reader, not the writer.)
- [ ] **Step 5:** Commit: `feat: heightmap export writers (png16/r16/exr/masks/metadata)`

### Task 6: Export UI + water material controls + post-regen audit

**Files:** Modify `src/main.rs`.

- [ ] **Step 1:** `CurrentHeightmap(Heightmap)` resource is filled by `poll_regen` (and `setup`). Panel "Export" section:

```rust
ui.collapsing("Export", |ui| {
    let busy = export_task.task.is_some();
    if busy {
        ui.spinner();
    } else if ui.button("Export heightmap + masks…").clicked() {
        if let Some(dir) = rfd::FileDialog::new().set_title("Export folder").pick_folder() {
            let hm = current_hm.0.clone();
            let params = params.clone();
            let frac = visual.sea_level_frac;
            export_task.task = Some(AsyncComputeTaskPool::get().spawn(async move {
                export::export_all(&hm, &params, frac, &dir)
                    .map(|files| format!("exported {} files to {}", files.len(), dir.display()))
            }));
        }
    }
    if let Some(msg) = &export_task.last_result {
        ui.label(msg);
    }
});
```

with `#[derive(Resource, Default)] struct ExportInFlight { task: Option<Task<Result<String, String>>>, last_result: Option<String> }` and a small poll system mirroring `poll_regen` (error strings shown in red via `ui.colored_label`).

- [ ] **Step 2:** Water material controls: store `water_material: Handle<StandardMaterial>` in `TerrainHandles`; Environment section gains an egui color button + roughness slider; `apply_visual_params` writes them through `Assets<StandardMaterial>::get_mut`. egui color: `let mut c = [r, g, b]; ui.color_edit_button_rgb(&mut c);` mapped back to `Color::srgba(c[0], c[1], c[2], 0.9)`.
- [ ] **Step 3:** Post-regen audit in `poll_regen`:

```rust
fn audit(hm: &Heightmap, tectonic_relief: f32, max_drop: f32) -> Result<(), String> {
    if !hm.data().iter().all(|v| v.is_finite()) {
        return Err("non-finite heights".into());
    }
    let spikes = crate::analysis::spike_mask(hm, max_drop * 2.0)
        .iter().filter(|&&b| b).count();
    let pct = spikes as f32 / hm.data().len() as f32;
    if pct > 0.005 {
        return Err(format!("spike density {:.2}% > 0.5%", pct * 100.0));
    }
    let (mn, mx) = /* min_max */;
    let relief = mx - mn;
    if relief < 0.5 * tectonic_relief || relief > 1.2 * tectonic_relief {
        return Err(format!("relief {relief:.1} outside [50%,120%] of tectonic {tectonic_relief:.1}"));
    }
    Ok(())
}
```

Result stored in `LastAudit(Result<(), String>)` resource; panel shows `✓ invariants ok` in green or the message in red.

- [ ] **Step 4:** Build, run, export to a temp folder, open `metadata.json` and `height_16.png` to confirm. `cargo nextest run` green.
- [ ] **Step 5:** Commit: `feat: export UI, water material controls, post-regen audit`

### Task 7: `tools/verify.ps1` + final verification

**Files:** Create `tools/verify.ps1`.

- [ ] **Step 1:**

```powershell
# TerraForge verification: full test suite + dual-viewpoint GPU captures.
# Run after ANY visually-affecting change, then compare the two PNGs against
# the previous pair from the same viewpoints before claiming success.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

cargo nextest run
if ($LASTEXITCODE -ne 0) { Write-Host "TESTS FAILED" -ForegroundColor Red; exit 1 }

cargo build --release
if ($LASTEXITCODE -ne 0) { exit 1 }

$env:BEVY_ASSET_ROOT = (Get-Location).Path

$env:TERRAFORGE_CAPTURE = "test_output/verify_close.png"
Remove-Item Env:TERRAFORGE_CAPTURE_VIEW -ErrorAction SilentlyContinue
& ./target/release/terraforge.exe

$env:TERRAFORGE_CAPTURE = "test_output/verify_wide.png"
$env:TERRAFORGE_CAPTURE_VIEW = "wide"
& ./target/release/terraforge.exe

Remove-Item Env:TERRAFORGE_CAPTURE, Env:TERRAFORGE_CAPTURE_VIEW -ErrorAction SilentlyContinue
Write-Host "OK — captures: test_output/verify_close.png, test_output/verify_wide.png" -ForegroundColor Green
```

- [ ] **Step 2:** Run `powershell -File tools/verify.ps1` — green, both captures produced. Visually compare both against the Phase 8 reference pair (defaults must be pixel-equivalent).
- [ ] **Step 3:** Commit: `feat: tools/verify.ps1 — one-command objective + subjective verification`

---

## Self-Review

**Spec coverage:** §1 map size → Task 3; §2 settings → Tasks 3, 6; §3 async → Task 4; §4 caching → Task 2; §5 export → Tasks 5, 6; §6 verification → Tasks 2 (bit-identity), 6 (audit), 7 (script); §7 boundaries → file structure. Error handling → Tasks 4 (task Err path keeps old terrain), 6 (export errors in panel). **Placeholders:** none — all code blocks complete; the one deliberate ellipsis (`min_max` reuse in audit) references a function defined in Task 5's module and trivially inlined. **Type consistency:** `PipelineParams`/`StageCache`/`PipelineRun`/`progress::*` names match across Tasks 2/4/5/6; `TerrainHandles` gains `water_material` (Task 6) and `water_mesh` (Task 3).

## Definition of done

- 40+ tests green including `incremental_equals_full`, 3 pipeline cache tests, 4 export tests.
- Talus-angle regen completes in well under 1 s at 2048²; UI stays interactive during full regens.
- Export folder opens in Unreal/Blender with correct scales (metadata.json).
- `tools/verify.ps1` passes; both captures at defaults match the Phase 8 reference pair.
