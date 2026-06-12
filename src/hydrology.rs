//! Static DEM hydrology: priority-flood depression filling (Barnes 2014,
//! epsilon variant) + D8 flow accumulation -> river network extraction,
//! river-bed carving and a water field (surface elevation + depth) for
//! rendering and export. Pure CPU, deterministic, no Bevy types.

use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;

use crate::heightmap::Heightmap;

#[derive(Clone, PartialEq, serde::Serialize)]
pub struct HydrologySettings {
    /// Minimum drainage area for a cell to count as river, as a fraction of
    /// total map cells (0.0002 ≈ dense network, 0.01 ≈ a few major rivers).
    pub river_threshold: f32,
    /// Max channel depth carved by the largest river, world units.
    pub carve_depth: f32,
    /// Channel half-width in cells for the largest river.
    pub river_width: f32,
}

impl Default for HydrologySettings {
    fn default() -> Self {
        Self { river_threshold: 0.001, carve_depth: 0.35, river_width: 2.0 }
    }
}

#[derive(Clone)]
pub struct WaterField {
    /// Water surface elevation; `f32::NEG_INFINITY` where dry.
    pub surface: Heightmap,
    /// Water depth (surface − bed); 0 where dry.
    pub depth: Heightmap,
}

pub struct HydrologyOutput {
    pub carved: Heightmap,
    pub water: WaterField,
    pub river_cells: usize,
    pub lake_cells: usize,
}

const NEIGHBOURS: [(i32, i32); 8] =
    [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)];

/// Min-heap entry (BinaryHeap is a max-heap; ordering reversed).
struct Cell {
    fill: f32,
    idx: usize,
}
impl PartialEq for Cell {
    fn eq(&self, o: &Self) -> bool {
        self.fill == o.fill && self.idx == o.idx
    }
}
impl Eq for Cell {}
impl PartialOrd for Cell {
    fn partial_cmp(&self, o: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cell {
    fn cmp(&self, o: &Self) -> CmpOrdering {
        o.fill
            .partial_cmp(&self.fill)
            .unwrap_or(CmpOrdering::Equal)
            // Tie-break on index for full determinism.
            .then_with(|| o.idx.cmp(&self.idx))
    }
}

/// Priority-flood with epsilon: returns the filled surface (>= terrain,
/// strictly draining to the border). Lakes are where `filled > terrain`.
pub fn priority_flood(hm: &Heightmap) -> Vec<f32> {
    let (w, h) = (hm.width, hm.height);
    let n = w * h;
    const EPS: f32 = 1e-5;
    let mut fill = vec![f32::NAN; n];
    let mut visited = vec![false; n];
    let mut open = BinaryHeap::with_capacity(2 * (w + h));

    let seed = |i: usize, open: &mut BinaryHeap<Cell>, visited: &mut Vec<bool>| {
        if !visited[i] {
            visited[i] = true;
            open.push(Cell { fill: hm.data()[i], idx: i });
        }
    };
    for x in 0..w {
        seed(x, &mut open, &mut visited);
        seed((h - 1) * w + x, &mut open, &mut visited);
    }
    for z in 0..h {
        seed(z * w, &mut open, &mut visited);
        seed(z * w + (w - 1), &mut open, &mut visited);
    }

    while let Some(Cell { fill: f, idx }) = open.pop() {
        fill[idx] = f;
        let x = (idx % w) as i32;
        let z = (idx / w) as i32;
        for (dx, dz) in NEIGHBOURS {
            let (nx, nz) = (x + dx, z + dz);
            if nx < 0 || nz < 0 || nx >= w as i32 || nz >= h as i32 {
                continue;
            }
            let j = nz as usize * w + nx as usize;
            if visited[j] {
                continue;
            }
            visited[j] = true;
            open.push(Cell { fill: hm.data()[j].max(f + EPS), idx: j });
        }
    }
    fill
}

/// D8 flow accumulation over the (epsilon-drained) filled surface: cells are
/// processed highest-first and each routes its drainage to the
/// steepest-descent neighbour. Returns drainage per cell, in cells (>= 1).
pub fn flow_accumulation(filled: &[f32], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        filled[b as usize]
            .partial_cmp(&filled[a as usize])
            .unwrap_or(CmpOrdering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let mut acc = vec![1.0f32; n];
    for &iu in &order {
        let i = iu as usize;
        let x = (i % w) as i32;
        let z = (i / w) as i32;
        let mut best: Option<usize> = None;
        let mut best_drop = 0.0f32;
        for (dx, dz) in NEIGHBOURS {
            let (nx, nz) = (x + dx, z + dz);
            if nx < 0 || nz < 0 || nx >= w as i32 || nz >= h as i32 {
                continue;
            }
            let j = nz as usize * w + nx as usize;
            let dist = if dx != 0 && dz != 0 { std::f32::consts::SQRT_2 } else { 1.0 };
            let drop = (filled[i] - filled[j]) / dist;
            if drop > best_drop {
                best_drop = drop;
                best = Some(j);
            }
        }
        if let Some(j) = best {
            acc[j] += acc[i];
        }
        // Border cells with no lower neighbour drain off-map.
    }
    acc
}

/// Full hydrology stage: fill -> accumulate -> carve channels -> water field.
pub fn run_hydrology(hm: &Heightmap, s: &HydrologySettings) -> HydrologyOutput {
    let (w, h) = (hm.width, hm.height);
    let n = w * h;
    let filled = priority_flood(hm);
    let acc = flow_accumulation(&filled, w, h);

    let thr = (s.river_threshold * n as f32).max(8.0);
    let max_acc = acc.iter().cloned().fold(0.0f32, f32::max).max(thr * 2.0);
    let log_thr = thr.ln();
    let log_span = (max_acc.ln() - log_thr).max(1e-6);

    // River strength field: radial parabolic channel profile around every
    // river cell, intensity scaled by log drainage (big rivers = deeper,
    // wider). `max` blending keeps confluences smooth.
    let mut strength = vec![0.0f32; n];
    for i in 0..n {
        if acc[i] < thr {
            continue;
        }
        let t = ((acc[i].ln() - log_thr) / log_span).clamp(0.0, 1.0);
        let radius = (1.0 + s.river_width * t).ceil() as i32;
        let x = (i % w) as i32;
        let z = (i / w) as i32;
        let r2 = (radius * radius) as f32;
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let (nx, nz) = (x + dx, z + dz);
                if nx < 0 || nz < 0 || nx >= w as i32 || nz >= h as i32 {
                    continue;
                }
                let d2 = (dx * dx + dz * dz) as f32;
                if d2 > r2 {
                    continue;
                }
                let j = nz as usize * w + nx as usize;
                let v = t * (1.0 - d2 / r2.max(1.0));
                if v > strength[j] {
                    strength[j] = v;
                }
            }
        }
    }

    let mut carved: Vec<f32> = hm.data().to_vec();
    let mut river_cells = 0usize;
    for i in 0..n {
        if strength[i] > 0.0 {
            carved[i] -= s.carve_depth * strength[i];
            river_cells += 1;
        }
    }

    // Water field: depressions stand at their fill level over the carved
    // bed; rivers run half-full in their channels.
    let mut surface = vec![f32::NEG_INFINITY; n];
    let mut depth = vec![0.0f32; n];
    let mut lake_cells = 0usize;
    for i in 0..n {
        if filled[i] > hm.data()[i] + 1e-4 {
            surface[i] = filled[i];
            depth[i] = filled[i] - carved[i];
            lake_cells += 1;
        } else if strength[i] > 0.0 {
            let water_h = carved[i] + 0.5 * s.carve_depth * strength[i];
            surface[i] = water_h;
            depth[i] = water_h - carved[i];
        }
    }

    HydrologyOutput {
        carved: Heightmap::from_fn(w, h, |x, z| carved[z * w + x]),
        water: WaterField {
            surface: Heightmap::from_fn(w, h, |x, z| surface[z * w + x]),
            depth: Heightmap::from_fn(w, h, |x, z| depth[z * w + x]),
        },
        river_cells,
        lake_cells,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16x16 bowl: rim at 10 with a notch at height 5, interior dips to 0.
    fn bowl() -> Heightmap {
        Heightmap::from_fn(16, 16, |x, z| {
            let edge = x == 0 || z == 0 || x == 15 || z == 15;
            if edge {
                if x == 8 && z == 0 {
                    5.0
                } else {
                    10.0
                }
            } else {
                let dx = x as f32 - 8.0;
                let dz = z as f32 - 8.0;
                (dx * dx + dz * dz).sqrt() * 0.5
            }
        })
    }

    #[test]
    fn fill_never_below_terrain() {
        let hm = bowl();
        let filled = priority_flood(&hm);
        for (i, &f) in filled.iter().enumerate() {
            assert!(f >= hm.data()[i] - 1e-6, "fill below terrain at {i}");
        }
    }

    #[test]
    fn bowl_fills_to_spill_notch() {
        let hm = bowl();
        let filled = priority_flood(&hm);
        let centre = filled[8 * 16 + 8];
        assert!(
            centre > 4.9 && centre < 5.2,
            "bowl centre filled to {centre}, expected ~5.0 (the notch height)"
        );
    }

    #[test]
    fn ramp_accumulates_downhill() {
        let hm = Heightmap::from_fn(16, 16, |_x, z| z as f32); // drains to z=0
        let filled = priority_flood(&hm);
        let acc = flow_accumulation(&filled, 16, 16);
        let top = acc[15 * 16 + 8];
        let bottom_max = (0..16).map(|x| acc[x]).fold(0.0f32, f32::max);
        assert!(bottom_max > top, "accumulation must grow downslope");
        assert!(bottom_max >= 14.0, "bottom row should drain its column, got {bottom_max}");
    }

    #[test]
    fn carve_is_bounded_and_localized() {
        let hm = Heightmap::from_fn(32, 32, |x, z| (x + z) as f32 * 0.2);
        let s = HydrologySettings::default();
        let out = run_hydrology(&hm, &s);
        let mut changed = 0usize;
        for i in 0..hm.data().len() {
            let delta = hm.data()[i] - out.carved.data()[i];
            assert!(delta >= -1e-6, "carving must never raise terrain");
            assert!(delta <= s.carve_depth + 1e-5, "carve exceeded max depth");
            if delta > 1e-6 {
                changed += 1;
            }
        }
        assert_eq!(changed > 0, out.river_cells > 0);
    }

    #[test]
    fn water_surface_above_bed_and_bowl_has_lake() {
        let hm = bowl();
        let out = run_hydrology(&hm, &HydrologySettings::default());
        for i in 0..hm.data().len() {
            let s = out.water.surface.data()[i];
            if s.is_finite() {
                assert!(out.water.depth.data()[i] > 0.0);
                assert!(s >= out.carved.data()[i] - 1e-4);
            }
        }
        assert!(out.lake_cells > 0, "bowl must contain a lake");
    }
}
