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
        assert!(
            hm.data().iter().any(|&h| (h - first).abs() > 1e-3),
            "fBm output should vary across the grid"
        );
    }

    #[test]
    fn heights_stay_within_amplitude_bounds() {
        let p = FbmParams { amplitude: 20.0, ..Default::default() };
        let hm = generate_fbm(96, 96, &p);
        // fBm normalized output is within ~[-1,1]; allow a small margin.
        let bound = p.amplitude * 1.2;
        assert!(
            hm.data().iter().all(|&h| h.abs() <= bound),
            "height exceeded {bound}"
        );
    }
}
