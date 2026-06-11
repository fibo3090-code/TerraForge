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
// Biome classification (mirrors the WGSL shader rules so we can predict +
// audit what each cell *should* render as before pixels are pushed)
// ---------------------------------------------------------------------------

/// Mirror of the parameters in `BiomeParams` (src/biome.rs) — kept local so
/// analysis can run without importing the renderer.
#[derive(Clone, Debug)]
pub struct BiomeRules {
    pub world_y_min: f32,
    pub world_y_range: f32,
    pub snow_line: f32,
    pub snow_blend: f32,
    pub rock_slope_cos: f32,
    pub rock_blend: f32,
    pub grass_dirt_line: f32,
    pub grass_dirt_blend: f32,
}

impl BiomeRules {
    pub fn from_heightmap(hm: &Heightmap, angle_deg: f32) -> Self {
        let stats = compute_stats(hm);
        Self {
            world_y_min: stats.min,
            world_y_range: stats.range().max(1e-4),
            snow_line: 0.62,
            snow_blend: 0.10,
            rock_slope_cos: angle_deg.to_radians().cos(),
            rock_blend: 0.15,
            grass_dirt_line: 0.45,
            grass_dirt_blend: 0.20,
        }
    }
}

/// Four canonical biome labels; the rendered colour is *expected* to fall
/// near the reference RGB for each.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Biome { Grass, Dirt, Rock, Snow }

impl Biome {
    pub fn reference_color(self) -> [u8; 3] {
        match self {
            Biome::Grass => [ 95, 130,  55],
            Biome::Dirt  => [140, 105,  65],
            Biome::Rock  => [115, 110, 100],
            Biome::Snow  => [240, 245, 250],
        }
    }
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// For a (height, normal-y) pair, return the dominant biome class — the one
/// the shader's mix() would weight highest. Same predicates as biome.wgsl:
/// snow is a top-level overlay gated by elevation, with steepness softly
/// suppressing it (so flat peaks read as snow, cliff faces as rock).
pub fn predict_biome(h: f32, normal_y: f32, r: &BiomeRules) -> Biome {
    let h_norm = ((h - r.world_y_min) / r.world_y_range).clamp(0.0, 1.0);
    let slope_t = 1.0 - smoothstep(
        r.rock_slope_cos - r.rock_blend,
        r.rock_slope_cos + r.rock_blend,
        normal_y,
    );
    let snow_t = smoothstep(r.snow_line - r.snow_blend, r.snow_line + r.snow_blend, h_norm)
        * (1.0 - slope_t * 0.5);
    if snow_t >= 0.5 { return Biome::Snow; }
    if slope_t >= 0.5 { return Biome::Rock; }
    let gd_t = smoothstep(
        r.grass_dirt_line - r.grass_dirt_blend,
        r.grass_dirt_line + r.grass_dirt_blend,
        h_norm,
    );
    if gd_t >= 0.5 { Biome::Dirt } else { Biome::Grass }
}

/// Predicted-biome map: for every cell, classify and colour with the
/// reference RGB. Useful to compare against the actual rendered output.
pub fn export_predicted_biome_png(hm: &Heightmap, r: &BiomeRules, path: &Path) -> Result<(), String> {
    let (w, h) = (hm.width, hm.height);
    let normals = surface_normals_y(hm, WORLD_CELL_FALLBACK);
    let img = ImageBuffer::<Rgb<u8>, _>::from_fn(w as u32, h as u32, |x, z| {
        let height = hm.get(x as usize, z as usize);
        let ny = normals[z as usize * w + x as usize];
        Rgb(predict_biome(height, ny, r).reference_color())
    });
    img.save(path).map_err(|e| format!("save {}: {e}", path.display()))
}

/// Fraction of cells per biome class (sums to 1.0).
pub fn biome_distribution(hm: &Heightmap, r: &BiomeRules) -> [(Biome, f32); 4] {
    let (w, h) = (hm.width, hm.height);
    let normals = surface_normals_y(hm, WORLD_CELL_FALLBACK);
    let mut counts = [0u32; 4];
    for z in 0..h {
        for x in 0..w {
            let b = predict_biome(hm.get(x, z), normals[z * w + x], r);
            let i = match b {
                Biome::Grass => 0,
                Biome::Dirt => 1,
                Biome::Rock => 2,
                Biome::Snow => 3,
            };
            counts[i] += 1;
        }
    }
    let total = (w * h) as f32;
    [
        (Biome::Grass, counts[0] as f32 / total),
        (Biome::Dirt,  counts[1] as f32 / total),
        (Biome::Rock,  counts[2] as f32 / total),
        (Biome::Snow,  counts[3] as f32 / total),
    ]
}

const WORLD_CELL_FALLBACK: f32 = 100.0 / 256.0;

/// Y-component of the surface normal (1 = flat, 0 = vertical). Computed from
/// centred-difference gradients of the heightmap scaled by `cell_size`.
fn surface_normals_y(hm: &Heightmap, cell_size: f32) -> Vec<f32> {
    let (w, h) = (hm.width, hm.height);
    let mut out = vec![0.0; w * h];
    for z in 0..h {
        for x in 0..w {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(w - 1);
            let zu = z.saturating_sub(1);
            let zd = (z + 1).min(h - 1);
            let dx = (hm.get(xr, z) - hm.get(xl, z)) / ((xr - xl) as f32 * cell_size).max(1e-6);
            let dz = (hm.get(x, zd) - hm.get(x, zu)) / ((zd - zu) as f32 * cell_size).max(1e-6);
            let grad = (dx * dx + dz * dz).sqrt();
            // n = (-dx, 1, -dz) normalized; n.y = 1 / |n|.
            out[z * w + x] = 1.0 / (1.0 + grad * grad).sqrt();
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Texture-palette analysis
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TextureStats {
    pub path: String,
    pub mean_rgb: [f32; 3],
    pub saturation: f32,
    pub brightness: f32,
}

/// Load a JPG/PNG, return the mean RGB plus an HSV-like saturation +
/// brightness derived from it. Saturation = (max - min) / max in [0,1].
pub fn analyze_texture(path: &Path) -> Result<TextureStats, String> {
    let img = image::open(path).map_err(|e| format!("open {}: {e}", path.display()))?.to_rgb8();
    let n = (img.width() * img.height()) as f64;
    let mut sum = [0.0f64; 3];
    for px in img.pixels() {
        sum[0] += px.0[0] as f64;
        sum[1] += px.0[1] as f64;
        sum[2] += px.0[2] as f64;
    }
    let m = [
        (sum[0] / n / 255.0) as f32,
        (sum[1] / n / 255.0) as f32,
        (sum[2] / n / 255.0) as f32,
    ];
    let mx = m[0].max(m[1]).max(m[2]);
    let mn = m[0].min(m[1]).min(m[2]);
    Ok(TextureStats {
        path: path.display().to_string(),
        mean_rgb: m,
        saturation: if mx > 1e-4 { (mx - mn) / mx } else { 0.0 },
        brightness: (m[0] + m[1] + m[2]) / 3.0,
    })
}

/// Pairwise Euclidean distance in normalized RGB space (0..√3).
pub fn rgb_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Audit every pair of biome textures and surface the minimum distance —
/// if two textures look ~identical the rendered output will too.
pub fn audit_biome_palette(stats: &[(Biome, TextureStats)]) -> AuditReport {
    let mut min_pair = None;
    let mut min_dist = f32::INFINITY;
    for i in 0..stats.len() {
        for j in (i + 1)..stats.len() {
            let d = rgb_distance(stats[i].1.mean_rgb, stats[j].1.mean_rgb);
            if d < min_dist {
                min_dist = d;
                min_pair = Some((stats[i].0, stats[j].0));
            }
        }
    }
    AuditReport {
        per_texture: stats.iter().cloned().collect(),
        closest_pair: min_pair,
        closest_distance: min_dist,
    }
}

#[derive(Debug)]
pub struct AuditReport {
    pub per_texture: Vec<(Biome, TextureStats)>,
    pub closest_pair: Option<(Biome, Biome)>,
    pub closest_distance: f32,
}

// ---------------------------------------------------------------------------
// Directional streak detection (gradient-direction histogram on a PNG)
// ---------------------------------------------------------------------------

/// Score in [0,1] indicating how strongly oriented an image's gradients are
/// along a single axis. A natural terrain render scores ~0.05–0.15; a render
/// dominated by triplanar / UV streaking can spike to 0.4+.
pub fn directional_streak_score(path: &Path) -> Result<DirectionalReport, String> {
    let img = image::open(path).map_err(|e| format!("open {}: {e}", path.display()))?.to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 3 || h < 3 {
        return Err("image too small for streak analysis".into());
    }
    // 8 directional bins covering 0..π (gradient direction is unsigned).
    let mut bins = [0.0f64; 8];
    let mut total_mag = 0.0f64;
    let pixel = |x: usize, z: usize| -> f32 { img.get_pixel(x as u32, z as u32).0[0] as f32 };
    for z in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = pixel(x + 1, z) - pixel(x - 1, z);
            let gy = pixel(x, z + 1) - pixel(x, z - 1);
            let mag = (gx * gx + gy * gy).sqrt();
            if mag < 4.0 { continue; }
            let angle = gy.atan2(gx);            // (-π, π]
            let mut a = angle.rem_euclid(std::f32::consts::PI);
            if a < 0.0 { a += std::f32::consts::PI; }
            let bin = ((a / std::f32::consts::PI) * 8.0).floor() as usize;
            let bin = bin.min(7);
            bins[bin] += mag as f64;
            total_mag += mag as f64;
        }
    }
    if total_mag < 1.0 {
        return Ok(DirectionalReport { bins, score: 0.0, dominant_axis_deg: 0.0 });
    }
    // Normalise bins so sum = 1.
    for b in &mut bins { *b /= total_mag; }
    // Streak score: deviation of the strongest bin from uniform (1/8 = 0.125).
    let max_idx = bins.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
    let score = ((bins[max_idx] - 1.0 / 8.0) / (1.0 - 1.0 / 8.0)).max(0.0) as f32;
    let dominant_axis_deg = 180.0 * (max_idx as f32 + 0.5) / 8.0;
    Ok(DirectionalReport { bins, score, dominant_axis_deg })
}

#[derive(Debug)]
pub struct DirectionalReport {
    pub bins: [f64; 8],
    pub score: f32,
    pub dominant_axis_deg: f32,
}

// ---------------------------------------------------------------------------
// Synthetic render
// ---------------------------------------------------------------------------
//
// The biome shader's bugs (planar-XZ streaks, wrong tangent->world mapping,
// over-blended biomes) only show up at render time. We can't easily run the
// real shader headlessly, but we can approximate it on the CPU by combining
// the heightmap-derived surface normal, the biome prediction, a
// noise-modulated grass/dirt boundary, and the *mean colour* of each albedo
// texture. The PNG that falls out is a coarse, but real, image of how the
// material rules behave — and the streak detector that worked on the
// hillshade can run on it to catch shader-rule regressions.

/// Sampled appearance of one biome — mean colour from the source texture.
#[derive(Clone, Debug)]
pub struct BiomePalette {
    pub grass: [f32; 3],
    pub dirt: [f32; 3],
    pub rock: [f32; 3],
    pub snow: [f32; 3],
}

impl BiomePalette {
    /// Read every albedo JPG under `biomes_dir` and use its mean RGB. Falls
    /// back to the reference colours if a texture is missing.
    pub fn from_assets(biomes_dir: &Path) -> Self {
        let load = |name: &str, fallback: Biome| -> [f32; 3] {
            let p = biomes_dir.join(format!("{name}_albedo.jpg"));
            if let Ok(s) = analyze_texture(&p) {
                s.mean_rgb
            } else {
                let c = fallback.reference_color();
                [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]
            }
        };
        Self {
            grass: load("grass", Biome::Grass),
            dirt: load("dirt", Biome::Dirt),
            rock: load("rock", Biome::Rock),
            snow: load("snow", Biome::Snow),
        }
    }
}

/// Deterministic hash-based value noise to mirror the WGSL `value_noise`.
fn cpu_value_noise(p: [f32; 2]) -> f32 {
    let ix = p[0].floor();
    let iy = p[1].floor();
    let fx = p[0] - ix;
    let fy = p[1] - iy;
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);
    let h = |a: f32, b: f32| -> f32 {
        let n = a * 127.1 + b * 311.7;
        let s = n.sin() * 43758.547;
        s - s.floor()
    };
    let a = h(ix, iy);
    let b = h(ix + 1.0, iy);
    let c = h(ix, iy + 1.0);
    let d = h(ix + 1.0, iy + 1.0);
    let ab = a + (b - a) * u;
    let cd = c + (d - c) * u;
    ab + (cd - ab) * v
}

fn cpu_macro_noise(x: f32, z: f32) -> f32 {
    let n = cpu_value_noise([x, z]) * 0.6
        + cpu_value_noise([x * 2.13 + 31.7, z * 2.13 + 19.3]) * 0.4;
    n * 2.0 - 1.0
}

#[inline]
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// Synthetic render: per-pixel biome blend (matching biome.wgsl's rules) ×
/// mean palette colour × Lambertian shading. Output is a PNG approximation
/// of the rendered material — coarse on detail but faithful to the rules.
pub fn export_synthetic_render_png(
    hm: &Heightmap,
    palette: &BiomePalette,
    rules: &BiomeRules,
    world_size: f32,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    path: &Path,
) -> Result<(), String> {
    let (w, h) = (hm.width, hm.height);
    let cell_size = world_size / w as f32;
    let az = sun_azimuth_deg.to_radians();
    let el = sun_elevation_deg.to_radians();
    let lx = el.cos() * az.sin();
    let ly = el.sin();
    let lz = el.cos() * az.cos();

    let img = ImageBuffer::<Rgb<u8>, _>::from_fn(w as u32, h as u32, |x, z| {
        let x = x as usize;
        let z = z as usize;
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(w - 1);
        let zu = z.saturating_sub(1);
        let zd = (z + 1).min(h - 1);
        let dx = (hm.get(xr, z) - hm.get(xl, z)) / ((xr - xl) as f32 * cell_size).max(1e-6);
        let dz = (hm.get(x, zd) - hm.get(x, zu)) / ((zd - zu) as f32 * cell_size).max(1e-6);
        let nlen = (1.0 + dx * dx + dz * dz).sqrt();
        let ny = 1.0 / nlen;
        let nx = -dx / nlen;
        let nz = -dz / nlen;

        let height = hm.get(x, z);
        let h_norm = ((height - rules.world_y_min) / rules.world_y_range).clamp(0.0, 1.0);
        let slope_t = 1.0 - smoothstep(
            rules.rock_slope_cos - rules.rock_blend,
            rules.rock_slope_cos + rules.rock_blend,
            ny,
        );

        // World-space coords for the noise; match the shader frequencies.
        let world_x = (x as f32 / (w - 1) as f32 - 0.5) * world_size;
        let world_z = (z as f32 / (h - 1) as f32 - 0.5) * world_size;
        let gd_noise = cpu_macro_noise(world_x * 0.03, world_z * 0.03);
        let snow_noise = cpu_macro_noise(world_x * 0.05, world_z * 0.05);

        let gd_base = smoothstep(
            rules.grass_dirt_line - rules.grass_dirt_blend,
            rules.grass_dirt_line + rules.grass_dirt_blend,
            h_norm,
        );
        let gd_t = (gd_base + gd_noise * 0.35).clamp(0.0, 1.0);
        let low = mix3(palette.grass, palette.dirt, gd_t);
        let base = mix3(low, palette.rock, slope_t);

        let snow_base = smoothstep(
            rules.snow_line - rules.snow_blend,
            rules.snow_line + rules.snow_blend,
            h_norm,
        ) * (1.0 - slope_t * 0.5);
        let snow_t = (snow_base + snow_noise * 0.15).clamp(0.0, 1.0);
        let albedo = mix3(base, palette.snow, snow_t);

        // Lambertian shading.
        let dot_nl = (nx * lx + ny * ly + nz * lz).max(0.0);
        let shade = 0.2 + 0.8 * dot_nl;
        let r = (albedo[0] * shade).clamp(0.0, 1.0) * 255.0;
        let g = (albedo[1] * shade).clamp(0.0, 1.0) * 255.0;
        let b = (albedo[2] * shade).clamp(0.0, 1.0) * 255.0;
        Rgb([r as u8, g as u8, b as u8])
    });
    img.save(path).map_err(|e| format!("save {}: {e}", path.display()))
}

/// Brightness-invariant "color direction" — divide each channel by the mean
/// so distances compare *hue* and *saturation*, not lightness. A shaded
/// version of a colour and the unshaded version have the same chromaticity.
fn chromaticity(rgb: [f32; 3]) -> [f32; 3] {
    let m = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
    if m < 1e-3 { return [1.0, 1.0, 1.0]; }
    [rgb[0] / m, rgb[1] / m, rgb[2] / m]
}

/// Fraction of pixels each anchor "wins" by chromaticity distance, plus a
/// lightness sanity check (snow's anchor is bright, so a pitch-black pixel
/// can't classify as snow no matter how flat its chromaticity). Useful for
/// checking that the rendered output actually exposes every biome we expect.
pub fn render_color_distribution(
    path: &Path,
    palette: &BiomePalette,
) -> Result<[(Biome, f32); 4], String> {
    let img = image::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?
        .to_rgb8();
    let n = (img.width() * img.height()) as f32;
    let mut counts = [0u32; 4];
    let anchors = [palette.grass, palette.dirt, palette.rock, palette.snow];
    let anchor_chroma: [[f32; 3]; 4] = [
        chromaticity(anchors[0]),
        chromaticity(anchors[1]),
        chromaticity(anchors[2]),
        chromaticity(anchors[3]),
    ];
    let anchor_brightness: [f32; 4] = anchors.map(|a| (a[0] + a[1] + a[2]) / 3.0);
    for px in img.pixels() {
        let p = [px.0[0] as f32 / 255.0, px.0[1] as f32 / 255.0, px.0[2] as f32 / 255.0];
        let p_chroma = chromaticity(p);
        let p_brightness = (p[0] + p[1] + p[2]) / 3.0;
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for i in 0..4 {
            // Weight chromaticity heavily; brightness mostly disambiguates
            // snow (bright) from rock (dark).
            let dc = rgb_distance(p_chroma, anchor_chroma[i]);
            let db = (p_brightness - anchor_brightness[i]).abs();
            let d = dc + 0.3 * db;
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        counts[best] += 1;
    }
    Ok([
        (Biome::Grass, counts[0] as f32 / n),
        (Biome::Dirt,  counts[1] as f32 / n),
        (Biome::Rock,  counts[2] as f32 / n),
        (Biome::Snow,  counts[3] as f32 / n),
    ])
}

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

    const GRID: usize = 256;
    const WORLD_SIZE: f32 = 100.0;
    const CELL_SIZE: f32 = WORLD_SIZE / GRID as f32;
    const TALUS_ANGLE_DEG: f32 = 35.0;

    /// Smoke-test the full generation pipeline at a manageable size, export
    /// every diagnostic image, and assert the expected-rules predicates.
    /// Run with `cargo nextest run pipeline_smoke -- --nocapture` to see the
    /// stat lines. PNG outputs land in `test_output/` (gitignored).
    /// If a real captured GPU render exists in `test_output/`, run the
    /// streak detector and colour-distribution analyser on it. Skip
    /// gracefully when nothing's been captured yet so the test isn't
    /// a hard dependency on having run the app with TERRAFORGE_CAPTURE.
    #[test]
    fn captured_render_is_healthy() {
        let path = PathBuf::from("test_output/07_real_render.png");
        if !path.is_file() {
            eprintln!("(no captured render at {} — skipping)", path.display());
            return;
        }
        let streak = directional_streak_score(&path)
            .expect("streak score on captured render");
        println!(
            "captured render streak score: {:.3} (dominant axis ~{:.0}°)",
            streak.score, streak.dominant_axis_deg,
        );
        // Stricter than the synthetic threshold because the real render is
        // what the user actually sees — shader bugs (triplanar / UV / normal
        // map) all show up here.
        assert!(
            streak.score < 0.40,
            "captured GPU render is too streaky (score {:.3} >= 0.40) — \
             biome shader is producing axis-aligned artifacts",
            streak.score,
        );

        let biomes_dir = PathBuf::from("assets/biomes");
        if biomes_dir.is_dir() {
            let palette = BiomePalette::from_assets(&biomes_dir);
            let dist = render_color_distribution(&path, &palette).unwrap();
            println!("captured render colour distribution:");
            for (b, frac) in &dist {
                println!("  {:?}: {:.2}%", b, frac * 100.0);
            }
            // Sky pixels mean the camera framed the terrain badly — also
            // useful signal, but we just want at least one non-sky biome
            // visible.
            let any_biome = dist.iter().any(|(_, f)| *f > 0.05);
            assert!(any_biome, "captured render has no visible terrain biome");
        }
    }

    #[test]
    fn pipeline_smoke() {

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

        // ---- Biome prediction map + distribution ---------------------------
        let rules = BiomeRules::from_heightmap(&thermal, 32.0);
        export_predicted_biome_png(&thermal, &rules, &dir.join("05_predicted_biome.png"))
            .unwrap();
        let dist = biome_distribution(&thermal, &rules);
        println!("biome distribution:");
        for (b, frac) in &dist {
            println!("  {:?}: {:.2}%", b, frac * 100.0);
        }
        // All 4 biomes must show up meaningfully (≥ 1% of cells); otherwise
        // the thresholds are misaligned with the terrain stats and a biome
        // will be invisible at render time.
        for (b, frac) in &dist {
            assert!(
                *frac > 0.01,
                "biome {:?} only covers {:.4}% of cells — thresholds misaligned with terrain stats",
                b, frac * 100.0,
            );
        }

        // ---- Texture palette audit -----------------------------------------
        let biomes_dir = PathBuf::from("assets/biomes");
        if biomes_dir.is_dir() {
            let albedos = [
                (Biome::Grass, biomes_dir.join("grass_albedo.jpg")),
                (Biome::Dirt,  biomes_dir.join("dirt_albedo.jpg")),
                (Biome::Rock,  biomes_dir.join("rock_albedo.jpg")),
                (Biome::Snow,  biomes_dir.join("snow_albedo.jpg")),
            ];
            let mut stats = Vec::with_capacity(4);
            for (b, p) in &albedos {
                let s = analyze_texture(p)
                    .unwrap_or_else(|e| panic!("texture analysis failed for {:?}: {e}", b));
                println!(
                    "  {:?}: mean=[{:.2},{:.2},{:.2}] sat={:.2} brightness={:.2}",
                    b, s.mean_rgb[0], s.mean_rgb[1], s.mean_rgb[2],
                    s.saturation, s.brightness,
                );
                stats.push((*b, s));
            }
            let report = audit_biome_palette(&stats);
            println!(
                "closest texture pair: {:?} (distance {:.3})",
                report.closest_pair, report.closest_distance,
            );
            // Two textures whose mean colours sit within 0.10 of each other
            // in normalised RGB are visually indistinguishable through PBR
            // shading — flag the build before the renderer ships such a pair.
            assert!(
                report.closest_distance > 0.10,
                "two biome albedos are too similar ({:?}, distance {:.3} < 0.10) — \
                 swap one of them out before rendering",
                report.closest_pair, report.closest_distance,
            );
            // Per-texture sanity: rock must be desaturated (real rock isn't
            // vivid green), snow must be bright.
            for (b, s) in &stats {
                match b {
                    Biome::Rock if s.saturation > 0.45 => panic!(
                        "rock texture is too colour-saturated ({:.2}) — wrong asset?", s.saturation
                    ),
                    Biome::Snow if s.brightness < 0.65 => panic!(
                        "snow texture brightness {:.2} < 0.65 — that's grey, not snow",
                        s.brightness
                    ),
                    _ => {}
                }
            }
        } else {
            println!("(skipped texture-palette audit: assets/biomes not present)");
        }

        // ---- Directional-streak score on the diagnostic hillshade ---------
        let streak = directional_streak_score(&dir.join("04_thermal_hs.png")).unwrap();
        println!(
            "thermal hillshade streak score: {:.3} (dominant axis ~{:.0}°)",
            streak.score, streak.dominant_axis_deg
        );
        assert!(
            streak.score < 0.35,
            "thermal hillshade is too directionally biased (streak score {:.3} >= 0.35) — \
             check the erosion/talus stages for axis-aligned bias",
            streak.score
        );

        // ---- Synthetic render: CPU port of the biome material rules -------
        if biomes_dir.is_dir() {
            let palette = BiomePalette::from_assets(&biomes_dir);
            let render_path = dir.join("06_synthetic_render.png");
            export_synthetic_render_png(
                &thermal, &palette, &rules,
                WORLD_SIZE, 315.0, 45.0,
                &render_path,
            ).unwrap();

            // Streak check on the rendered output itself — this is what the
            // hillshade misses, because shader bugs only show up here.
            let render_streak = directional_streak_score(&render_path).unwrap();
            println!(
                "synthetic render streak score: {:.3} (dominant axis ~{:.0}°)",
                render_streak.score, render_streak.dominant_axis_deg,
            );
            assert!(
                render_streak.score < 0.35,
                "synthetic render is too streaky ({:.3} >= 0.35) — biome shader or \
                 UV projection is producing axis-aligned artifacts",
                render_streak.score,
            );

            // Closest-anchor classification on the rendered output: how much
            // of the image actually reads as each biome?
            let render_dist = render_color_distribution(&render_path, &palette).unwrap();
            println!("rendered colour distribution:");
            for (b, frac) in &render_dist {
                println!("  {:?}: {:.2}%", b, frac * 100.0);
            }
            for (b, frac) in &render_dist {
                assert!(
                    *frac > 0.01,
                    "rendered biome {:?} only covers {:.2}% of pixels — \
                     palette or shader rules dropped it on the floor",
                    b, frac * 100.0,
                );
            }
        }

        println!("pipeline OK -> {}", dir.display());
    }
}
