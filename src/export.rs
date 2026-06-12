//! Pure heightmap -> file writers. No Bevy types; everything is testable
//! headless. All errors are surfaced as Err(String) (spec: no silent export
//! failures).

use std::io::Write;
use std::path::Path;

use crate::analysis::BiomeRules;
use crate::heightmap::Heightmap;
use crate::pipeline::{min_max, PipelineParams, METERS_PER_UNIT};

/// 16-bit grayscale PNG, min–max normalized. Returns the (min, max) used so
/// callers can record the scale.
pub fn write_png16(hm: &Heightmap, path: &Path) -> Result<(f32, f32), String> {
    let (mn, mx) = min_max(hm);
    let range = (mx - mn).max(1e-6);
    let img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_fn(
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
    let mut f =
        std::fs::File::create(path).map_err(|e| format!("r16 {}: {e}", path.display()))?;
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
    let img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_fn(
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
    let files: [(&str, Result<(), String>); 6] = [
        ("height_16.png", write_png16(hm, &dir.join("height_16.png")).map(|_| ())),
        ("height.r16", write_r16(hm, &dir.join("height.r16")).map(|_| ())),
        ("height.exr", write_exr(hm, &dir.join("height.exr"))),
        ("biome_mask.png", write_biome_mask(hm, &rules, &dir.join("biome_mask.png"))),
        (
            "water_depth_16.png",
            write_water_depth16(hm, sea_level, &dir.join("water_depth_16.png")),
        ),
        ("metadata.json", write_metadata(hm, params, sea_level_frac, &dir.join("metadata.json"))),
    ];
    let mut written = Vec::new();
    for (name, result) in files {
        result?;
        written.push(name.to_string());
    }
    Ok(written)
}

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
        let path = tmp_dir("png16").join("h.png");
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
        let path = tmp_dir("exr").join("h.exr");
        write_exr(&hm, &path).unwrap();
        let img = exr::prelude::read_first_rgba_layer_from_file(
            &path,
            |size, _| vec![0.0f32; size.width() * size.height()],
            |buf: &mut Vec<f32>, pos, (r, _g, _b, _a): (f32, f32, f32, f32)| {
                buf[pos.y() * 32 + pos.x()] = r;
            },
        )
        .unwrap();
        for z in 0..32 {
            for x in 0..32 {
                assert_eq!(
                    img.layer_data.channel_data.pixels[z * 32 + x],
                    hm.get(x, z),
                    "mismatch at ({x},{z})"
                );
            }
        }
    }

    #[test]
    fn r16_length_and_endianness() {
        let hm = test_hm();
        let path = tmp_dir("r16").join("h.r16");
        let (mn, mx) = write_r16(&hm, &path).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 32 * 32 * 2);
        let first = u16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 65535.0;
        let v = mn + first * (mx - mn);
        assert!((v - hm.get(0, 0)).abs() <= (mx - mn) / 65535.0 + 1e-4);
    }

    #[test]
    fn metadata_json_contains_scale_and_params() {
        let hm = test_hm();
        let path = tmp_dir("meta").join("metadata.json");
        let params = crate::pipeline::PipelineParams::default();
        write_metadata(&hm, &params, 0.12, &path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(v["height_min_units"].is_number());
        assert!(v["world_size_meters"].is_number());
        assert_eq!(v["params"]["base"]["seed"], 0);
    }

    #[test]
    fn export_all_writes_six_files() {
        let hm = test_hm();
        let dir = tmp_dir("all");
        let files =
            export_all(&hm, &crate::pipeline::PipelineParams::default(), 0.12, &dir).unwrap();
        assert_eq!(files.len(), 6);
        for f in &files {
            assert!(dir.join(f).is_file(), "{f} missing");
        }
    }
}
