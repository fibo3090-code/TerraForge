//! CPU reference implementation of the talus / angle-of-repose pass.
//! Phase 5 moved the production pipeline to `erosion::erode_thermal` (GPU);
//! this module is retained as an algorithmic reference and an independent
//! cross-check for the GPU stage (its 4 unit tests still run on every build).
//!
//! Not on the hot path — `apply_talus` and `TalusParams` are flagged dead-code
//! tolerant rather than re-exported. Phase 6 can delete this file outright if
//! no one is reading it.

#![allow(dead_code)]

use crate::heightmap::Heightmap;

/// Parameters for [`apply_talus`].
#[derive(Clone, Debug)]
pub struct TalusParams {
    /// Max permitted height drop between 4-neighbors (in heightmap units).
    /// Any neighbor pair exceeding this loses excess equally between the two.
    pub max_drop: f32,
    /// Fraction of the excess moved per iteration (0..1). Smaller = gentler.
    pub damping: f32,
    /// Number of relaxation iterations.
    pub iterations: u32,
}

impl Default for TalusParams {
    fn default() -> Self {
        Self { max_drop: 0.35, damping: 0.5, iterations: 4 }
    }
}

/// Apply a few relaxation iterations: cells higher than a neighbor by more
/// than `max_drop` shed material to that neighbor (and vice versa). Mass is
/// conserved to f32 rounding because each redistribution is symmetric.
pub fn apply_talus(hm: &Heightmap, params: &TalusParams) -> Heightmap {
    let (w, h) = (hm.width, hm.height);
    let mut a: Vec<f32> = hm.data().to_vec();
    let mut b: Vec<f32> = a.clone();
    for _ in 0..params.iterations {
        for z in 0..h {
            for x in 0..w {
                let i = z * w + x;
                let me = a[i];
                let mut delta = 0.0;
                let mut push = |nx: usize, nz: usize| {
                    let n = a[nz * w + nx];
                    let diff = n - me; // positive: neighbor higher than me
                    if diff > params.max_drop {
                        // Neighbor is too tall: receive half the excess.
                        delta += (diff - params.max_drop) * params.damping * 0.5;
                    } else if diff < -params.max_drop {
                        // I'm too tall: shed half the excess.
                        delta += (diff + params.max_drop) * params.damping * 0.5;
                    }
                };
                if x > 0       { push(x - 1, z); }
                if x + 1 < w   { push(x + 1, z); }
                if z > 0       { push(x, z - 1); }
                if z + 1 < h   { push(x, z + 1); }
                // Each push moves half the excess. Mass is conserved pair-wise
                // (donor loses what receiver gains), so no extra scaling needed.
                b[i] = me + delta;
            }
        }
        std::mem::swap(&mut a, &mut b);
    }
    Heightmap::from_fn(w, h, |x, z| a[z * w + x])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_heightmap_is_unchanged() {
        let hm = Heightmap::new(8, 8);
        let out = apply_talus(&hm, &TalusParams::default());
        assert_eq!(out.data(), hm.data());
    }

    #[test]
    fn single_spike_is_lowered() {
        let mut hm = Heightmap::new(5, 5);
        hm.set(2, 2, 10.0);
        let out = apply_talus(&hm, &TalusParams::default());
        assert!(out.get(2, 2) < 10.0, "spike must be smoothed");
        // Neighbors must gain some height (mass conservation).
        assert!(out.get(1, 2) > 0.0);
        assert!(out.get(3, 2) > 0.0);
        assert!(out.get(2, 1) > 0.0);
        assert!(out.get(2, 3) > 0.0);
    }

    #[test]
    fn conserves_mass() {
        let hm = Heightmap::from_fn(16, 16, |x, z| ((x * 7 + z * 13) % 11) as f32 * 0.3);
        let out = apply_talus(&hm, &TalusParams::default());
        let sum_in: f32 = hm.data().iter().sum();
        let sum_out: f32 = out.data().iter().sum();
        assert!((sum_in - sum_out).abs() < 1e-3, "talus must conserve mass ({sum_in} -> {sum_out})");
    }

    #[test]
    fn gentle_slope_is_preserved() {
        // A 0.1-per-cell ramp (well under default max_drop=0.35) should not move.
        let hm = Heightmap::from_fn(8, 8, |x, _z| x as f32 * 0.1);
        let out = apply_talus(&hm, &TalusParams::default());
        for (a, b) in out.data().iter().zip(hm.data()) {
            assert!((a - b).abs() < 1e-5, "ramp below talus must stay put");
        }
    }
}
