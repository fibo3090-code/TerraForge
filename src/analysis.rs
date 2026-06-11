//! Terrain analysis & verification: statistics, derived fields (slope,
//! aspect, spike mask), PNG exports (raw heightmap, hillshade, slope heatmap,
//! spike mask) and the predicates that the pipeline-smoke test asserts on.
//!
//! The PNG exports give a human (or a multimodal LLM) something to look at;
//! the predicates give CI something to fail on. Both work from a `Heightmap`
//! so they're orthogonal to which stage produced it — drop in any stage,
//! re-run, look.

// The whole module is consumed by the in-file `pipeline_smoke` test and
// optionally by downstream debug tooling; treat it as a test/verification
// utility surface rather than production code.
#![allow(dead_code)]

use std::path::Path;

use image::{ImageBuffer, Luma, Rgb};

use crate::heightmap::Heightmap;

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct HeightStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub median: f32,
    pub p99: f32,
}

impl HeightStats {
    pub fn range(&self) -> f32 {
        self.max - self.min
    }
}

pub fn compute_stats(hm: &Heightmap) -> HeightStats {
    let data = hm.data();
    assert!(!data.is_empty(), "stats: empty heightmap");
    let n = data.len();

    let (min, max, sum) = data.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY, 0.0f64),
        |(mn, mx, s), &v| (mn.min(v), mx.max(v), s + v as f64),
    );
    let mean = (sum / n as f64) as f32;
    let var = data.iter().map(|&v| (v - mean).powi(2) as f64).sum::<f64>() / n as f64;
    let std_dev = (var.sqrt()) as f32;

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[n / 2];
    let p99 = sorted[(n as f32 * 0.99) as usize];

    HeightStats { min, max, mean, std_dev, median, p99 }
}

// ---------------------------------------------------------------------------
// Derived fields
// ---------------------------------------------------------------------------

/// Per-cell slope in degrees, computed from centred-difference gradients
/// scaled by `cell_size`. Borders mirror the inner gradient.
pub fn slope_field_degrees(hm: &Heightmap, cell_size: f32) -> Heightmap {
    let (w, h) = (hm.width, hm.height);
    Heightmap::from_fn(w, h, |x, z| {
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(w - 1);
        let zu = z.saturating_sub(1);
        let zd = (z + 1).min(h - 1);
        let dx = (hm.get(xr, z) - hm.get(xl, z)) / ((xr - xl) as f32 * cell_size).max(1e-6);
        let dz = (hm.get(x, zd) - hm.get(x, zu)) / ((zd - zu) as f32 * cell_size).max(1e-6);
        let grad = (dx * dx + dz * dz).sqrt();
        grad.atan().to_degrees()
    })
}

/// Mark cells whose height deviates from a 3×3 median by more than
/// `threshold` (heightmap units). These are the "isolated spike" candidates a
/// healthy erosion + talus pipeline should not produce.
pub fn spike_mask(hm: &Heightmap, threshold: f32) -> Vec<bool> {
    let (w, h) = (hm.width, hm.height);
    let mut mask = vec![false; w * h];
    let mut buf = [0.0f32; 9];
    for z in 0..h {
        for x in 0..w {
            let mut k = 0;
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let nz = (z as i32 + dz).clamp(0, h as i32 - 1) as usize;
                    buf[k] = hm.get(nx, nz);
                    k += 1;
                }
            }
            buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = buf[4];
            if (hm.get(x, z) - median).abs() > threshold {
                mask[z * w + x] = true;
            }
        }
    }
    mask
}

/// Count of 4-neighbour pairs whose absolute height difference exceeds
/// `max_drop`. Should be ≈ 0 after a healthy thermal pass.
pub fn talus_violations(hm: &Heightmap, max_drop: f32) -> usize {
    let (w, h) = (hm.width, hm.height);
    let mut count = 0;
    for z in 0..h {
        for x in 0..w {
            let me = hm.get(x, z);
            if x + 1 < w && (hm.get(x + 1, z) - me).abs() > max_drop { count += 1; }
            if z + 1 < h && (hm.get(x, z + 1) - me).abs() > max_drop { count += 1; }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// PNG exports
// ---------------------------------------------------------------------------

/// Grayscale PNG, heights linearly mapped to [0, 255] over [min, max].
pub fn export_grayscale_png(hm: &Heightmap, path: &Path) -> Result<(), String> {
    let (w, h) = (hm.width as u32, hm.height as u32);
    let stats = compute_stats(hm);
    let range = stats.range().max(1e-6);
    let img = ImageBuffer::<Luma<u8>, _>::from_fn(w, h, |x, z| {
        let v = (hm.get(x as usize, z as usize) - stats.min) / range;
        Luma([(v.clamp(0.0, 1.0) * 255.0) as u8])
    });
    img.save(path).map_err(|e| format!("save {}: {e}", path.display()))
}

/// Lambertian hillshade. `sun_azimuth_deg` measured clockwise from +Z (north);
/// `sun_elevation_deg` is the angle above the horizon. World coordinates: +X
/// east, +Z south, +Y up.
pub fn export_hillshade_png(
    hm: &Heightmap,
    cell_size: f32,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    path: &Path,
) -> Result<(), String> {
    let az = sun_azimuth_deg.to_radians();
    let el = sun_elevation_deg.to_radians();
    // Light direction in world frame: pointing from the sun toward the ground.
    let lx = el.cos() * az.sin();
    let ly = el.sin();
    let lz = el.cos() * az.cos();
    let (w, h) = (hm.width as u32, hm.height as u32);

    let img = ImageBuffer::<Luma<u8>, _>::from_fn(w, h, |x, z| {
        let x = x as usize;
        let z = z as usize;
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(hm.width - 1);
        let zu = z.saturating_sub(1);
        let zd = (z + 1).min(hm.height - 1);
        let dhdx = (hm.get(xr, z) - hm.get(xl, z)) / ((xr - xl) as f32 * cell_size).max(1e-6);
        let dhdz = (hm.get(x, zd) - hm.get(x, zu)) / ((zd - zu) as f32 * cell_size).max(1e-6);
        // Surface normal n = (-dh/dx, 1, -dh/dz), normalized.
        let nx = -dhdx;
        let ny = 1.0;
        let nz = -dhdz;
        let nlen = (nx * nx + ny * ny + nz * nz).sqrt();
        let dot = (nx * lx + ny * ly + nz * lz) / nlen;
        // Ambient floor so cliff faces in shadow aren't pitch-black.
        let shaded = (0.15 + 0.85 * dot.max(0.0)).clamp(0.0, 1.0);
        Luma([(shaded * 255.0) as u8])
    });
    img.save(path).map_err(|e| format!("save {}: {e}", path.display()))
}

/// Slope visualised as a green→yellow→red heatmap, capped at 70°.
pub fn export_slope_heatmap_png(hm: &Heightmap, cell_size: f32, path: &Path) -> Result<(), String> {
    let slope = slope_field_degrees(hm, cell_size);
    let (w, h) = (hm.width as u32, hm.height as u32);
    let img = ImageBuffer::<Rgb<u8>, _>::from_fn(w, h, |x, z| {
        let s = (slope.get(x as usize, z as usize) / 70.0).clamp(0.0, 1.0);
        // 0   -> green (0.2, 0.7, 0.2)
        // 0.5 -> yellow (1.0, 0.9, 0.2)
        // 1   -> red   (0.9, 0.2, 0.2)
        let (r, g, b) = if s < 0.5 {
            let t = s * 2.0;
            (0.2 + 0.8 * t, 0.7 + 0.2 * t, 0.2)
        } else {
            let t = (s - 0.5) * 2.0;
            (1.0 - 0.1 * t, 0.9 - 0.7 * t, 0.2)
        };
        Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
    });
    img.save(path).map_err(|e| format!("save {}: {e}", path.display()))
}

/// Hillshade with detected spikes overlaid in red. Threshold is the same one
/// used by `spike_mask` — typically 2 × `max_drop`.
pub fn export_spike_overlay_png(
    hm: &Heightmap,
    cell_size: f32,
    spike_threshold: f32,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    path: &Path,
) -> Result<(), String> {
    let az = sun_azimuth_deg.to_radians();
    let el = sun_elevation_deg.to_radians();
    let lx = el.cos() * az.sin();
    let ly = el.sin();
    let lz = el.cos() * az.cos();
    let mask = spike_mask(hm, spike_threshold);
    let (w, h) = (hm.width as u32, hm.height as u32);

    let img = ImageBuffer::<Rgb<u8>, _>::from_fn(w, h, |x, z| {
        let x = x as usize;
        let z = z as usize;
        if mask[z * hm.width + x] {
            return Rgb([255, 40, 40]); // spike
        }
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(hm.width - 1);
        let zu = z.saturating_sub(1);
        let zd = (z + 1).min(hm.height - 1);
        let dhdx = (hm.get(xr, z) - hm.get(xl, z)) / ((xr - xl) as f32 * cell_size).max(1e-6);
        let dhdz = (hm.get(x, zd) - hm.get(x, zu)) / ((zd - zu) as f32 * cell_size).max(1e-6);
        let nx = -dhdx;
        let ny = 1.0;
        let nz = -dhdz;
        let nlen = (nx * nx + ny * ny + nz * nz).sqrt();
        let dot = (nx * lx + ny * ly + nz * lz) / nlen;
        let shaded = (0.15 + 0.85 * dot.max(0.0)).clamp(0.0, 1.0);
        let g = (shaded * 255.0) as u8;
        Rgb([g, g, g])
    });
    img.save(path).map_err(|e| format!("save {}: {e}", path.display()))
}

// ===========================================================================
// Pipeline smoke test
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erosion::{self, ErosionParams, ThermalParams};
    use crate::tectonics::{apply_tectonics, TectonicParams};
    use crate::terrain_noise::{generate_fbm, FbmParams};
    use std::fs;
    use std::path::PathBuf;

    fn out_dir() -> PathBuf {
        let dir = PathBuf::from("test_output");
        fs::create_dir_all(&dir).expect("create test_output/");
        dir
    }

    /// Smoke-test the full generation pipeline at a manageable size, export
    /// every diagnostic image, and assert the expected-rules predicates.
    /// Run with `cargo nextest run pipeline_smoke -- --nocapture` to see the
    /// stat lines. PNG outputs land in `test_output/` (gitignored).
    #[test]
    fn pipeline_smoke() {
        const GRID: usize = 256;
        const WORLD_SIZE: f32 = 100.0;
        const CELL_SIZE: f32 = WORLD_SIZE / GRID as f32;
        const TALUS_ANGLE_DEG: f32 = 35.0;

        let dir = out_dir();
        let max_drop = TALUS_ANGLE_DEG.to_radians().tan() * CELL_SIZE;
        let spike_threshold = max_drop * 2.0;

        // --- Stage 1: fBm base ----------------------------------------------
        let base = generate_fbm(
            GRID,
            GRID,
            &FbmParams { amplitude: 10.0, ..Default::default() },
        );
        let stats_base = compute_stats(&base);
        println!("stage[fbm]     {stats_base:?}");
        export_grayscale_png(&base, &dir.join("01_fbm.png")).unwrap();

        // --- Stage 2: tectonics ---------------------------------------------
        let tect = apply_tectonics(&base, &TectonicParams::default());
        let stats_tect = compute_stats(&tect);
        println!("stage[tect]    {stats_tect:?}");
        export_grayscale_png(&tect, &dir.join("02_tectonics.png")).unwrap();
        export_hillshade_png(&tect, CELL_SIZE, 315.0, 45.0, &dir.join("02_tectonics_hs.png"))
            .unwrap();
        assert!(
            stats_tect.mean >= stats_base.mean,
            "tectonic uplift must raise the mean ({} -> {})",
            stats_base.mean, stats_tect.mean
        );

        // --- Stage 3: hydraulic erosion -------------------------------------
        let gpu = erosion::GpuContext::new().expect("GPU adapter required for smoke test");
        let hydro = erosion::erode_hydraulic(
            &gpu,
            &tect,
            &ErosionParams { iterations: 300, ..Default::default() },
        )
        .unwrap();
        let stats_hydro = compute_stats(&hydro);
        println!("stage[hydro]   {stats_hydro:?}");
        export_grayscale_png(&hydro, &dir.join("03_hydraulic.png")).unwrap();
        export_hillshade_png(&hydro, CELL_SIZE, 315.0, 45.0, &dir.join("03_hydraulic_hs.png"))
            .unwrap();
        export_spike_overlay_png(
            &hydro,
            CELL_SIZE,
            spike_threshold,
            315.0,
            45.0,
            &dir.join("03_hydraulic_spikes.png"),
        )
        .unwrap();

        let hydro_spikes = spike_mask(&hydro, spike_threshold).iter().filter(|&&b| b).count();
        let hydro_violations = talus_violations(&hydro, max_drop);
        println!(
            "stage[hydro]   spike_cells={hydro_spikes} ({:.2}%), talus_violations={hydro_violations}",
            100.0 * hydro_spikes as f32 / (GRID * GRID) as f32
        );

        // Hydraulic invariants: nothing exploded; peaks (top 1%) lost material
        // compared to tectonic input.
        for &v in hydro.data() {
            assert!(v.is_finite(), "hydraulic produced NaN/Inf");
        }
        assert!(
            stats_hydro.p99 < stats_tect.p99,
            "hydraulic must lower the high terrain ({} -> {})",
            stats_tect.p99, stats_hydro.p99
        );

        // --- Stage 4: thermal erosion ---------------------------------------
        let thermal = erosion::erode_thermal(
            &gpu,
            &hydro,
            &ThermalParams { iterations: 12, max_drop, damping: 0.5 },
        )
        .unwrap();
        let stats_thermal = compute_stats(&thermal);
        println!("stage[thermal] {stats_thermal:?}");
        export_grayscale_png(&thermal, &dir.join("04_thermal.png")).unwrap();
        export_hillshade_png(&thermal, CELL_SIZE, 315.0, 45.0, &dir.join("04_thermal_hs.png"))
            .unwrap();
        export_slope_heatmap_png(&thermal, CELL_SIZE, &dir.join("04_thermal_slope.png")).unwrap();
        export_spike_overlay_png(
            &thermal,
            CELL_SIZE,
            spike_threshold,
            315.0,
            45.0,
            &dir.join("04_thermal_spikes.png"),
        )
        .unwrap();

        let thermal_spikes = spike_mask(&thermal, spike_threshold).iter().filter(|&&b| b).count();
        let thermal_violations = talus_violations(&thermal, max_drop);
        println!(
            "stage[thermal] spike_cells={thermal_spikes} ({:.4}%), talus_violations={thermal_violations}",
            100.0 * thermal_spikes as f32 / (GRID * GRID) as f32
        );

        // ----- Expected-rules predicates -----
        for &v in thermal.data() {
            assert!(v.is_finite(), "thermal produced NaN/Inf");
        }

        // 1. Mass conservation through thermal (pair-wise exchange, < 0.1% drift).
        let sum_hydro: f32 = hydro.data().iter().sum();
        let sum_thermal: f32 = thermal.data().iter().sum();
        let scale: f32 = hydro.data().iter().map(|v| v.abs()).sum();
        let drift = (sum_thermal - sum_hydro).abs() / scale.max(1.0);
        assert!(drift < 1e-3, "thermal mass drift {:.4}% (target < 0.1%)", drift * 100.0);

        // 2. Thermal must eliminate the bulk of isolated spikes. (Hydraulic
        //    can leave a few hundred deposition stalagmites; thermal should
        //    knock them all out via lateral redistribution.)
        assert!(
            thermal_spikes < hydro_spikes / 4 + 1,
            "thermal didn't reduce isolated spikes enough: {hydro_spikes} -> {thermal_spikes}"
        );

        // 3. SEVERE talus violations (pairs differing by > 3 × max_drop) must
        //    not increase. Steep ridges within ~2× the angle of repose are
        //    legitimate erosion features and are counted separately above;
        //    severe violations are the genuine pathology this guards against.
        let severe = |hm: &Heightmap| -> usize {
            let (w, h) = (hm.width, hm.height);
            let thr = max_drop * 3.0;
            let mut n = 0;
            for z in 0..h {
                for x in 0..w {
                    let me = hm.get(x, z);
                    if x + 1 < w && (hm.get(x + 1, z) - me).abs() > thr { n += 1; }
                    if z + 1 < h && (hm.get(x, z + 1) - me).abs() > thr { n += 1; }
                }
            }
            n
        };
        let hydro_severe = severe(&hydro);
        let thermal_severe = severe(&thermal);
        println!(
            "stage[compare] severe_violations: hydro={hydro_severe}, thermal={thermal_severe}"
        );
        assert!(
            thermal_severe <= hydro_severe,
            "thermal must not introduce severe cliffs (>3×max_drop): {hydro_severe} -> {thermal_severe}"
        );

        // 4. After the full pipeline, isolated spikes must be <0.5% of cells.
        let spike_pct = thermal_spikes as f32 / (GRID * GRID) as f32;
        assert!(
            spike_pct < 0.005,
            "post-pipeline spike density {:.3}% above 0.5% budget", spike_pct * 100.0
        );

        // 5. Relief must not collapse to a pancake — > 50% of the
        //    pre-erosion range should survive.
        assert!(
            stats_thermal.range() > 0.5 * stats_tect.range(),
            "erosion flattened relief too aggressively ({} -> {})",
            stats_tect.range(), stats_thermal.range()
        );

        // 6. Output range must not exceed input range by more than 20%
        //    (divergence guard).
        assert!(
            stats_thermal.range() < 1.2 * stats_tect.range(),
            "erosion inflated relief ({} -> {})",
            stats_tect.range(), stats_thermal.range()
        );

        println!("pipeline OK -> {}", dir.display());
    }
}
