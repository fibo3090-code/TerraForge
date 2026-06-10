//! CPU-side heightmap: an N×N grid of f32 heights.
//! Row-major: index = z * width + x. Every generation/erosion stage in the
//! pipeline consumes and produces this type (the GPU texture mirror is added
//! in the erosion phase).

/// A row-major grid of heights.
#[derive(Clone, Debug)]
pub struct Heightmap {
    pub width: usize,
    pub height: usize,
    data: Vec<f32>,
}

impl Heightmap {
    /// A `width`×`height` heightmap filled with zeros.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0.0; width * height] }
    }

    /// Build a heightmap by sampling `f(x, z)` over the grid.
    pub fn from_fn(width: usize, height: usize, f: impl Fn(usize, usize) -> f32) -> Self {
        let mut hm = Self::new(width, height);
        for z in 0..height {
            for x in 0..width {
                hm.set(x, z, f(x, z));
            }
        }
        hm
    }

    #[inline]
    pub fn get(&self, x: usize, z: usize) -> f32 {
        self.data[z * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, z: usize, value: f32) {
        self.data[z * self.width + x] = value;
    }

    /// Raw row-major heights, length `width * height`.
    /// Used by tests now; consumed by GPU texture upload in the erosion phase.
    #[allow(dead_code)]
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_dims_and_is_zeroed() {
        let hm = Heightmap::new(4, 3);
        assert_eq!(hm.width, 4);
        assert_eq!(hm.height, 3);
        assert_eq!(hm.data().len(), 12);
        assert!(hm.data().iter().all(|&h| h == 0.0));
    }

    #[test]
    fn from_fn_indexes_row_major() {
        // value encodes (x, z) so we can verify indexing direction
        let hm = Heightmap::from_fn(3, 2, |x, z| (z * 3 + x) as f32);
        assert_eq!(hm.get(0, 0), 0.0);
        assert_eq!(hm.get(2, 0), 2.0);
        assert_eq!(hm.get(0, 1), 3.0);
        assert_eq!(hm.get(2, 1), 5.0);
    }

    #[test]
    fn set_then_get_roundtrips() {
        let mut hm = Heightmap::new(2, 2);
        hm.set(1, 0, 4.2);
        assert_eq!(hm.get(1, 0), 4.2);
        assert_eq!(hm.get(0, 0), 0.0);
    }
}
