use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

mod heightmap;
mod mesh_builder;

use heightmap::Heightmap;
use mesh_builder::heightmap_to_mesh;

const GRID: usize = 128;
const WORLD_SIZE: f32 = 100.0;
const HEIGHT_SCALE: f32 = 1.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Trivial placeholder heightmap: two crossed cosine waves -> a gentle hill
    // field, just to prove generation -> mesh -> render. Real noise is Phase 2.
    let hm = Heightmap::from_fn(GRID, GRID, |x, z| {
        let fx = x as f32 / (GRID - 1) as f32;
        let fz = z as f32 / (GRID - 1) as f32;
        let h = (fx * std::f32::consts::TAU).cos() + (fz * std::f32::consts::TAU).cos();
        h * 6.0
    });
    let mesh = heightmap_to_mesh(&hm, WORLD_SIZE, HEIGHT_SCALE);

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.45, 0.42, 0.38),
            perceptual_roughness: 0.95,
            ..default()
        })),
    ));

    // Sun
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(50.0, 80.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Fly-around (orbit) camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 60.0, 110.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
