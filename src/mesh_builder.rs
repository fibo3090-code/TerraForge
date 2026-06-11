//! Builds a renderable Bevy mesh from a `Heightmap`.
//! Grid is centred on the origin in the XZ plane; height is +Y.

use bevy::asset::RenderAssetUsages;
use bevy::math::Vec3;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};

use crate::heightmap::Heightmap;

/// Convert a heightmap into a triangle-list mesh with positions, recomputed
/// normals, and UVs. `world_size` is the X/Z extent in world units; heights
/// are multiplied by `height_scale`.
pub fn heightmap_to_mesh(hm: &Heightmap, world_size: f32, height_scale: f32) -> Mesh {
    let (w, h) = (hm.width, hm.height);
    assert!(w >= 2 && h >= 2, "heightmap must be at least 2x2 to build a mesh");

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(w * h);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(w * h);

    for z in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32; // 0..=1
            let fz = z as f32 / (h - 1) as f32;
            let px = (fx - 0.5) * world_size;
            let pz = (fz - 0.5) * world_size;
            let py = hm.get(x, z) * height_scale;
            positions.push([px, py, pz]);
            uvs.push([fx, fz]);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity((w - 1) * (h - 1) * 6);
    for z in 0..h - 1 {
        for x in 0..w - 1 {
            let i = (z * w + x) as u32;
            let right = i + 1;
            let down = i + w as u32;
            let down_right = down + 1;
            // Winding chosen so flat ground normals point +Y (see compute_normals).
            indices.extend_from_slice(&[i, down, right]);
            indices.extend_from_slice(&[right, down, down_right]);
        }
    }

    let normals = compute_normals(&positions, &indices);
    let colors = slope_height_colors(&positions, &normals);

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

/// Per-vertex RGBA shading band. Pure presentation — no effect on the
/// simulation. Three regions blended by elevation, then a steepness mask
/// pushes any slope above ~25° toward rock so erosion channels read clearly:
///   - low (h < 0.30):  meadow / dry grass
///   - mid (h ~ 0.55):  dirt
///   - high (h > 0.80): rock, with snow above the snow line on gentle faces
fn slope_height_colors(positions: &[[f32; 3]], normals: &[[f32; 3]]) -> Vec<[f32; 4]> {
    let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in positions {
        y_min = y_min.min(p[1]);
        y_max = y_max.max(p[1]);
    }
    let y_range = (y_max - y_min).max(1e-6);

    let grass = Vec3::new(0.36, 0.42, 0.22);
    let dirt = Vec3::new(0.52, 0.42, 0.28);
    let rock = Vec3::new(0.48, 0.43, 0.39);
    let snow = Vec3::new(0.96, 0.97, 1.00);

    positions
        .iter()
        .zip(normals)
        .map(|(p, n)| {
            let h = ((p[1] - y_min) / y_range).clamp(0.0, 1.0);
            // Smooth elevation band: grass -> dirt -> rock.
            let low_to_mid = smoothstep(0.18, 0.45, h);
            let mid_to_high = smoothstep(0.55, 0.82, h);
            let elev = grass.lerp(dirt, low_to_mid).lerp(rock, mid_to_high);

            // Slope steepness in [0,1]: 0 = flat, 1 = vertical face.
            let steepness = (1.0 - n[1]).clamp(0.0, 1.0);
            // Anything above 25°-ish reads as exposed rock.
            let rock_blend = smoothstep(0.18, 0.55, steepness);
            let base = elev.lerp(rock, rock_blend);

            // Snow on gentle high ground only.
            let snow_t = smoothstep(0.62, 0.92, h) * (1.0 - smoothstep(0.10, 0.35, steepness));
            let c = base.lerp(snow, snow_t);
            [c.x, c.y, c.z, 1.0]
        })
        .collect()
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Area-weighted vertex normals from face normals.
fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let a = Vec3::from_array(positions[tri[0] as usize]);
        let b = Vec3::from_array(positions[tri[1] as usize]);
        let c = Vec3::from_array(positions[tri[2] as usize]);
        let face = (b - a).cross(c - a); // length ∝ triangle area
        for &idx in tri {
            normals[idx as usize] += face;
        }
    }
    normals
        .iter()
        .map(|n| n.normalize_or_zero().to_array())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    fn positions_of(mesh: &Mesh) -> Vec<[f32; 3]> {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("positions not Float32x3"),
        }
    }
    fn normals_of(mesh: &Mesh) -> Vec<[f32; 3]> {
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap() {
            VertexAttributeValues::Float32x3(v) => v.clone(),
            _ => panic!("normals not Float32x3"),
        }
    }

    #[test]
    fn vertex_and_index_counts_are_correct() {
        let hm = Heightmap::new(4, 3); // flat
        let mesh = heightmap_to_mesh(&hm, 10.0, 1.0);
        assert_eq!(positions_of(&mesh).len(), 4 * 3);
        let n_indices = match mesh.indices().unwrap() {
            Indices::U32(v) => v.len(),
            Indices::U16(v) => v.len(),
        };
        assert_eq!(n_indices, (4 - 1) * (3 - 1) * 6);
    }

    #[test]
    fn flat_heightmap_normals_point_up() {
        let hm = Heightmap::new(5, 5); // all zero
        let mesh = heightmap_to_mesh(&hm, 8.0, 3.0);
        for n in normals_of(&mesh) {
            assert!((n[0]).abs() < 1e-5, "nx={} not ~0", n[0]);
            assert!((n[1] - 1.0).abs() < 1e-5, "ny={} not ~1 (winding wrong?)", n[1]);
            assert!((n[2]).abs() < 1e-5, "nz={} not ~0", n[2]);
        }
    }

    #[test]
    fn grid_is_centred_and_spans_world_size() {
        let hm = Heightmap::new(3, 3);
        let world = 6.0_f32;
        let mesh = heightmap_to_mesh(&hm, world, 1.0);
        let p = positions_of(&mesh);
        // first vertex (x=0,z=0) is the min corner, last is the max corner
        assert_eq!(p[0], [-world / 2.0, 0.0, -world / 2.0]);
        assert_eq!(*p.last().unwrap(), [world / 2.0, 0.0, world / 2.0]);
    }

    #[test]
    fn height_maps_to_y_with_scale() {
        let hm = Heightmap::from_fn(2, 2, |x, _z| if x == 1 { 2.0 } else { 0.0 });
        let mesh = heightmap_to_mesh(&hm, 4.0, 5.0);
        let p = positions_of(&mesh);
        // vertex at x=0 -> y 0; vertex at x=1 -> y = 2.0 * 5.0
        assert_eq!(p[0][1], 0.0);
        assert_eq!(p[1][1], 10.0);
    }
}
