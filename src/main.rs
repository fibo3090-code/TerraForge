use bevy::anti_alias::fxaa::Fxaa;
use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{
    light_consts::lux, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder,
    DirectionalLightShadowMap, GlobalAmbientLight, NotShadowCaster,
};
use bevy::pbr::{Atmosphere, AtmosphereSettings, MaterialPlugin, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

mod analysis;
mod biome;
mod erosion;
mod heightmap;
mod mesh_builder;
mod talus;
mod tectonics;
mod terrain_noise;

use biome::{build_terrain_material, BiomeParams, TerrainMaterial};
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

/// Optional capture mode: when `TERRAFORGE_CAPTURE=path/to/out.png` is set,
/// the app renders a few frames, screenshots the primary window to that path,
/// and exits. Used by the verification harness to feed the *real* GPU render
/// into the streak / colour-distribution analysers.
#[derive(Resource)]
struct CaptureMode {
    output_path: String,
    frames_waited: u32,
    triggered: bool,
}

fn main() {
    let mut app = App::new();
    // Sky + ambient both come from the atmosphere now: the render-sky node
    // draws the background and AtmosphereEnvironmentMapLight supplies the
    // sky-bounce IBL, so the flat clear colour and hand-tuned ambient go.
    app.insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
        .add_systems(Startup, setup);

    if let Ok(path) = std::env::var("TERRAFORGE_CAPTURE") {
        app.insert_resource(CaptureMode {
            output_path: path,
            frames_waited: 0,
            triggered: false,
        });
        app.add_systems(Update, capture_then_exit);
    }
    app.run();
}

/// Wait long enough for the material to compile + a few frames to settle,
/// then take a screenshot and request app exit on the following frame.
fn capture_then_exit(
    mut commands: Commands,
    mut capture: ResMut<CaptureMode>,
    mut exit: MessageWriter<AppExit>,
) {
    capture.frames_waited += 1;
    // Step 1: take the screenshot after ~120 frames (covers material compile,
    // texture upload, and one stable render).
    if !capture.triggered && capture.frames_waited >= 120 {
        info!("capture: writing screenshot to {}", capture.output_path);
        commands.spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.output_path.clone()));
        capture.triggered = true;
    }
    // Step 2: once triggered, give the screenshot system a few frames to
    // finish writing to disk, then exit cleanly.
    if capture.triggered && capture.frames_waited >= 150 {
        info!("capture: exiting");
        exit.write(AppExit::Success);
    }
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
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

    // Biome material reads the post-erosion height range so the elevation
    // smoothsteps line up with this seed's actual relief.
    let biome_params = BiomeParams {
        world_y_min: h_min,
        world_y_range: (h_max - h_min).max(1e-3),
        ..Default::default()
    };
    let terrain_mat = build_terrain_material(&asset_server, &mut materials, &mut images, biome_params);
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(terrain_mat),
    ));

    // Water plane: floods the lowest valleys. Sea level is derived from this
    // seed's stats rather than hard-coded so re-tuning erosion never leaves
    // the water floating above (or buried under) the terrain.
    let water_level = h_min + 0.12 * (h_max - h_min);
    commands.spawn((
        // 20x the terrain so the plane's edge stays beyond the horizon line
        // from any sane orbit position.
        Mesh3d(meshes.add(Plane3d::default().mesh().size(WORLD_SIZE * 20.0, WORLD_SIZE * 20.0))),
        MeshMaterial3d(std_materials.add(StandardMaterial {
            base_color: Color::srgba(0.08, 0.22, 0.32, 0.9),
            perceptual_roughness: 0.08,
            metallic: 0.0,
            reflectance: 0.4,
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, water_level, 0.0),
        NotShadowCaster,
    ));
    info!("water plane at y={water_level:.2}");

    // Sun at physical illuminance — the atmosphere LUTs expect real-world
    // lux, and the camera's Exposure(ev100=13) brings it back to display
    // range. Cascaded shadows sized to the terrain extent.
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
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

    // Fly-around (orbit) camera with MSAA and tone mapping. Capture mode
    // forces a close-up perspective looking down a slope — that's where
    // shader streak artifacts are loudest, so the autonomous capture has
    // the best chance of catching them.
    // Capture viewpoints: "close" (default) frames a slope where texture
    // artifacts are loudest; "wide" frames the whole terrain + horizon so sky
    // and water coverage can be verified. Always capture BOTH when validating
    // a visual change — artifacts hide at the viewpoint you didn't check.
    let view = std::env::var("TERRAFORGE_CAPTURE_VIEW").unwrap_or_default();
    let (cam_pos, cam_target) = if std::env::var("TERRAFORGE_CAPTURE").is_ok() {
        if view == "wide" {
            (Vec3::new(0.0, 55.0, 150.0), Vec3::new(0.0, 5.0, 0.0))
        } else {
            (Vec3::new(15.0, h_max as f32 + 8.0, 15.0), Vec3::new(-25.0, h_mean as f32, -25.0))
        }
    } else {
        (Vec3::new(0.0, 70.0, 130.0), Vec3::ZERO)
    };
    commands.spawn((
        Camera3d::default(),
        // Physically-based sky + aerial perspective. scene_units_to_m = 20
        // reads the 100-unit terrain as a ~2 km massif: enough air for subtle
        // haze on far ridges without fogging the whole scene (100 here put
        // ~15 km between the orbit camera and the terrain and washed it out).
        Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
        AtmosphereSettings { scene_units_to_m: 20.0, ..default() },
        // Sky-driven ambient + reflections for this view.
        AtmosphereEnvironmentMapLight::default(),
        Exposure { ev100: 13.0 },
        Bloom::NATURAL,
        // The atmosphere render-sky node is incompatible with MSAA; the
        // upstream example pairs it with FXAA instead.
        Msaa::Off,
        Fxaa::default(),
        Tonemapping::AcesFitted,
        Transform::from_xyz(cam_pos.x, cam_pos.y, cam_pos.z).looking_at(cam_target, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
