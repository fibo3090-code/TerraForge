use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

mod heightmap;
mod mesh_builder;
mod tectonics;
mod terrain_noise;

use mesh_builder::heightmap_to_mesh;
use tectonics::{apply_tectonics, TectonicParams};
use terrain_noise::{generate_fbm, FbmParams};

const GRID: usize = 256;
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
    // Pipeline: fBm base -> tectonic uplift + ridge guidance.
    let base = generate_fbm(
        GRID,
        GRID,
        &FbmParams { amplitude: 10.0, ..Default::default() },
    );
    let hm = apply_tectonics(&base, &TectonicParams::default());
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
