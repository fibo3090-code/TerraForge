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
