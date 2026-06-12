//! Cached four-stage generation pipeline:
//! fBm base -> tectonics -> hydraulic erosion -> thermal erosion.
//! Each stage's cache entry stores the exact params that produced it, so a
//! param change re-runs only the affected suffix of the cascade. The
//! `incremental_equals_full` test guarantees cached runs are bit-identical
//! to cold runs.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::erosion::{self, GpuContext};
use crate::heightmap::Heightmap;
use crate::tectonics::{apply_tectonics, TectonicParams};
use crate::terrain_noise::{generate_fbm, FbmParams};

pub const TALUS_ITERATIONS: u32 = 12;
/// Render scale: 1 world unit = 20 m (drives km/m display + export metadata).
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

    pub fn label(stage: u8) -> &'static str {
        ["fBm", "tectonics", "hydraulic", "thermal", "meshing", "done"]
            .get(stage as usize)
            .unwrap_or(&"…")
    }
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

pub fn min_max(hm: &Heightmap) -> (f32, f32) {
    hm.data()
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)))
}

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
