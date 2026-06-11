use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{CascadeShadowConfigBuilder, DirectionalLightShadowMap, GlobalAmbientLight};
use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

mod analysis;
mod erosion;
mod heightmap;
mod mesh_builder;
mod talus;
mod tectonics;
mod terrain_noise;

use mesh_builder::heightmap_to_mesh;
use tectonics::{apply_tectonics, TectonicParams};
use terrain_noise::{generate_fbm, FbmParams};

const GRID: usize = 2048;
const WORLD_SIZE: f32 = 100.0;
const HEIGHT_SCALE: f32 = 1.0;
/// Linear erosion budget per cell side: at 512² we ran 800 iterations, so
/// keep the per-cell step density similar at higher grids.
const EROSION_ITERATIONS: u32 = (GRID as u32 * 3) / 4;
/// Target angle of repose for the talus pass. Real dry talus is 30–40°.
const TALUS_ANGLE_DEG: f32 = 35.0;
const TALUS_ITERATIONS: u32 = 12;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.55, 0.68, 0.82)))
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.75, 0.82, 0.95),
            brightness: 2500.0,
            ..default()
        })
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
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
    // Pipeline: fBm base -> tectonic uplift + ridges -> GPU hydraulic
    // erosion -> CPU talus relaxation -> mesh.
    let t = std::time::Instant::now();
    let base = generate_fbm(
        GRID,
        GRID,
        &FbmParams { amplitude: 10.0, ..Default::default() },
    );
    info!("fbm base ({GRID}x{GRID}): {:?}", t.elapsed());

    let t = std::time::Instant::now();
    let hm = apply_tectonics(&base, &TectonicParams::default());
    info!("tectonic stage: {:?}", t.elapsed());

    let gpu = erosion::GpuContext::new()
        .expect("GPU compute unavailable - hydraulic erosion requires a wgpu adapter");
    let t = std::time::Instant::now();
    let hm = erosion::erode_hydraulic(
        &gpu,
        &hm,
        &erosion::ErosionParams {
            iterations: EROSION_ITERATIONS,
            ..Default::default()
        },
    )
    .expect("hydraulic erosion failed");
    info!(
        "hydraulic erosion ({EROSION_ITERATIONS} iter at {GRID}x{GRID}): {:?}",
        t.elapsed()
    );

    // max permitted height drop between neighbours = tan(angle) * cell_size.
    // Scales naturally with resolution so the visual angle stays constant.
    let cell_size = WORLD_SIZE / GRID as f32;
    let max_drop = TALUS_ANGLE_DEG.to_radians().tan() * cell_size;
    let t = std::time::Instant::now();
    let hm = erosion::erode_thermal(
        &gpu,
        &hm,
        &erosion::ThermalParams {
            iterations: TALUS_ITERATIONS,
            max_drop,
            damping: 0.5,
        },
    )
    .expect("thermal erosion failed");
    info!(
        "thermal erosion (max_drop={max_drop:.3}, {TALUS_ITERATIONS} iter): {:?}",
        t.elapsed()
    );

    // Sanity check: erosion should not push the field outside the bounds
    // already enforced by the GPU divergence clamp and tectonic uplift range.
    let (h_min, h_max, h_sum) =
        hm.data().iter().fold((f32::INFINITY, f32::NEG_INFINITY, 0.0f64), |(mn, mx, s), &v| {
            (mn.min(v), mx.max(v), s + v as f64)
        });
    let h_mean = h_sum / hm.data().len() as f64;
    info!(
        "heightmap stats: min={h_min:.2}, max={h_max:.2}, mean={h_mean:.2}, range={:.2}",
        h_max - h_min
    );
    assert!(h_min.is_finite() && h_max.is_finite(), "erosion produced NaN/Inf");

    let t = std::time::Instant::now();
    let mesh = heightmap_to_mesh(&hm, WORLD_SIZE, HEIGHT_SCALE);
    info!("mesh build ({} verts): {:?}", GRID * GRID, t.elapsed());

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            // White base lets the per-vertex slope/height colours show through.
            base_color: Color::WHITE,
            perceptual_roughness: 0.92,
            reflectance: 0.05,
            ..default()
        })),
    ));

    // Warm sun + cascaded shadows that actually frame the terrain extent.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.96, 0.88),
            illuminance: 11_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(40.0, 120.0, 60.0).looking_at(Vec3::ZERO, Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            first_cascade_far_bound: 30.0,
            maximum_distance: 400.0,
            ..default()
        }
        .build(),
    ));

    // Fly-around (orbit) camera with MSAA and tone mapping.
    commands.spawn((
        Camera3d::default(),
        Msaa::Sample4,
        Tonemapping::TonyMcMapface,
        Transform::from_xyz(0.0, 70.0, 130.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
