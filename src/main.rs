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
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

mod analysis;
mod biome;
mod erosion;
mod heightmap;
mod mesh_builder;
mod pipeline;
mod talus;
mod tectonics;
mod terrain_noise;

use biome::{build_terrain_material, BiomeParams, TerrainMaterial};
use heightmap::Heightmap;
use mesh_builder::heightmap_to_mesh;
use tectonics::{apply_tectonics, TectonicParams};
use terrain_noise::{generate_fbm, FbmParams};

const WORLD_SIZE: f32 = 100.0;
const HEIGHT_SCALE: f32 = 1.0;
const TALUS_ITERATIONS: u32 = 12;

// ---------------------------------------------------------------------------
// Parameters (edited live from the egui panel)
// ---------------------------------------------------------------------------

/// Parameters that require a full pipeline re-run (seconds). Applied when the
/// user presses "Regenerate" in the panel.
#[derive(Resource, Clone, PartialEq)]
struct GenParams {
    seed: u32,
    /// Grid resolution per side. 512 regenerates in ~0.3 s for fast
    /// iteration; 2048 is presentation quality (~4 s).
    grid: usize,
    /// fBm base amplitude in world units.
    amplitude: f32,
    /// Tectonic uplift peak height in world units.
    uplift_strength: f32,
    /// Ridge crest height in fully-uplifted zones.
    ridge_strength: f32,
    /// Target angle of repose for the thermal stage.
    talus_angle_deg: f32,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            seed: 0,
            grid: 2048,
            amplitude: 10.0,
            uplift_strength: 14.0,
            ridge_strength: 16.0,
            talus_angle_deg: 35.0,
        }
    }
}

/// Parameters that apply instantly (uniform / transform mutation only).
#[derive(Resource, Clone, PartialEq)]
struct VisualParams {
    /// Sea level as a fraction of the heightmap range above its minimum.
    sea_level_frac: f32,
    /// Normalized elevation of the snow band centre.
    snow_line: f32,
    /// Slope angle (degrees) past which terrain reads as rock.
    rock_slope_deg: f32,
    /// Normalized elevation of the grass->dirt transition.
    grass_dirt_line: f32,
    /// Sun spherical angles, degrees.
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
}

impl Default for VisualParams {
    fn default() -> Self {
        Self {
            sea_level_frac: 0.12,
            snow_line: 0.62,
            rock_slope_deg: 32.0,
            grass_dirt_line: 0.45,
            // Matches the Phase 7 sun at (40, 120, 60).
            sun_azimuth_deg: 33.7,
            sun_elevation_deg: 59.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

/// The headless compute context survives regenerations.
#[derive(Resource)]
struct GpuCompute(erosion::GpuContext);

#[derive(Resource, Clone, Copy)]
struct TerrainStats {
    h_min: f32,
    h_max: f32,
}

impl TerrainStats {
    fn range(&self) -> f32 {
        (self.h_max - self.h_min).max(1e-3)
    }
    fn water_level(&self, frac: f32) -> f32 {
        self.h_min + frac * self.range()
    }
}

/// Asset handles that live across regenerations: the mesh asset is replaced
/// in place, the material's uniform is mutated.
#[derive(Resource)]
struct TerrainHandles {
    mesh: Handle<Mesh>,
    material: Handle<TerrainMaterial>,
}

#[derive(Resource, Default)]
struct RegenRequested(bool);

/// Wall-clock duration of the last pipeline run, displayed in the panel.
#[derive(Resource, Default)]
struct LastRegenSecs(f32);

#[derive(Component)]
struct WaterPlane;

#[derive(Component)]
struct Sun;

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
    // Sky + ambient both come from the atmosphere: the render-sky node draws
    // the background and AtmosphereEnvironmentMapLight supplies sky-bounce IBL.
    app.insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .insert_resource(GenParams::default())
        .insert_resource(VisualParams::default())
        .insert_resource(RegenRequested::default())
        .insert_resource(LastRegenSecs::default())
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin::default())
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ui_panel)
        .add_systems(Update, (apply_visual_params, regenerate_terrain));

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
    if !capture.triggered && capture.frames_waited >= 120 {
        info!("capture: writing screenshot to {}", capture.output_path);
        commands.spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.output_path.clone()));
        capture.triggered = true;
    }
    if capture.triggered && capture.frames_waited >= 150 {
        info!("capture: exiting");
        exit.write(AppExit::Success);
    }
}

// ---------------------------------------------------------------------------
// Generation pipeline
// ---------------------------------------------------------------------------

/// fBm base -> tectonic uplift + ridges -> GPU hydraulic erosion -> GPU
/// thermal erosion. Returns the heightmap plus its min/max.
fn run_pipeline(gpu: &erosion::GpuContext, p: &GenParams) -> (Heightmap, TerrainStats) {
    let grid = p.grid;
    let t_total = std::time::Instant::now();

    let base = generate_fbm(
        grid,
        grid,
        &FbmParams { seed: p.seed, amplitude: p.amplitude, ..Default::default() },
    );
    let hm = apply_tectonics(
        &base,
        &TectonicParams {
            seed: p.seed,
            uplift_strength: p.uplift_strength,
            ridge_strength: p.ridge_strength,
            ..Default::default()
        },
    );

    // Linear erosion budget per cell side, same density at every resolution.
    let erosion_iterations = (grid as u32 * 3) / 4;
    let hm = erosion::erode_hydraulic(
        gpu,
        &hm,
        &erosion::ErosionParams { iterations: erosion_iterations, ..Default::default() },
    )
    .expect("hydraulic erosion failed");

    // Max permitted neighbour drop = tan(angle) * cell_size, so the visual
    // angle of repose stays constant across grid resolutions.
    let cell_size = WORLD_SIZE / grid as f32;
    let max_drop = p.talus_angle_deg.to_radians().tan() * cell_size;
    let hm = erosion::erode_thermal(
        gpu,
        &hm,
        &erosion::ThermalParams { iterations: TALUS_ITERATIONS, max_drop, damping: 0.5 },
    )
    .expect("thermal erosion failed");

    let (h_min, h_max) = hm
        .data()
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    assert!(h_min.is_finite() && h_max.is_finite(), "erosion produced NaN/Inf");
    info!(
        "pipeline: {grid}x{grid}, seed {}, {erosion_iterations} erosion iter, \
         heights [{h_min:.2}, {h_max:.2}] in {:?}",
        p.seed,
        t_total.elapsed()
    );

    (hm, TerrainStats { h_min, h_max })
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gen_params: Res<GenParams>,
    visual: Res<VisualParams>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let gpu = erosion::GpuContext::new()
        .expect("GPU compute unavailable - hydraulic erosion requires a wgpu adapter");
    let (hm, stats) = run_pipeline(&gpu, &gen_params);
    commands.insert_resource(GpuCompute(gpu));
    commands.insert_resource(stats);

    let mesh = heightmap_to_mesh(&hm, WORLD_SIZE, HEIGHT_SCALE);
    let mesh_handle = meshes.add(mesh);

    // Biome material reads the post-erosion height range so the elevation
    // smoothsteps line up with this seed's actual relief.
    let biome_params = BiomeParams {
        world_y_min: stats.h_min,
        world_y_range: stats.range(),
        snow_line: visual.snow_line,
        rock_slope_cos: visual.rock_slope_deg.to_radians().cos(),
        grass_dirt_line: visual.grass_dirt_line,
        ..Default::default()
    };
    let material_handle =
        build_terrain_material(&asset_server, &mut materials, &mut images, biome_params);
    commands.insert_resource(TerrainHandles {
        mesh: mesh_handle.clone(),
        material: material_handle.clone(),
    });
    commands.spawn((Mesh3d(mesh_handle), MeshMaterial3d(material_handle)));

    // Water plane: floods the lowest valleys. Sea level derives from the
    // heightmap stats so erosion re-tuning never strands it.
    let water_level = stats.water_level(visual.sea_level_frac);
    commands.spawn((
        // 20x the terrain so the plane's edge stays beyond the horizon.
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
        WaterPlane,
    ));

    // Sun at physical illuminance — the atmosphere LUTs expect real-world
    // lux; the camera's Exposure brings it back to display range.
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        sun_transform(visual.sun_azimuth_deg, visual.sun_elevation_deg),
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            first_cascade_far_bound: 30.0,
            maximum_distance: 400.0,
            ..default()
        }
        .build(),
        Sun,
    ));

    // Capture viewpoints: "close" (default) frames a slope where texture
    // artifacts are loudest; "wide" frames the whole terrain + horizon.
    // Always capture BOTH when validating a visual change.
    let view = std::env::var("TERRAFORGE_CAPTURE_VIEW").unwrap_or_default();
    let (cam_pos, cam_target) = if std::env::var("TERRAFORGE_CAPTURE").is_ok() {
        if view == "wide" {
            (Vec3::new(0.0, 55.0, 150.0), Vec3::new(0.0, 5.0, 0.0))
        } else {
            let mid = (stats.h_min + stats.h_max) * 0.5;
            (Vec3::new(15.0, stats.h_max + 8.0, 15.0), Vec3::new(-25.0, mid, -25.0))
        }
    } else {
        (Vec3::new(0.0, 70.0, 130.0), Vec3::ZERO)
    };
    commands.spawn((
        Camera3d::default(),
        // Physically-based sky + aerial perspective. scene_units_to_m = 20
        // reads the 100-unit terrain as a ~2 km massif: subtle haze on far
        // ridges without fogging the scene.
        Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
        AtmosphereSettings { scene_units_to_m: 20.0, ..default() },
        AtmosphereEnvironmentMapLight::default(),
        Exposure { ev100: 13.0 },
        Bloom::NATURAL,
        // The atmosphere render-sky node is incompatible with MSAA; pair
        // with FXAA per the upstream example.
        Msaa::Off,
        Fxaa::default(),
        Tonemapping::AcesFitted,
        Transform::from_xyz(cam_pos.x, cam_pos.y, cam_pos.z).looking_at(cam_target, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}

fn sun_transform(azimuth_deg: f32, elevation_deg: f32) -> Transform {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.to_radians();
    let dir = Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos());
    Transform::from_translation(dir * 200.0).looking_at(Vec3::ZERO, Vec3::Y)
}

// ---------------------------------------------------------------------------
// Live controls
// ---------------------------------------------------------------------------

fn ui_panel(
    mut contexts: EguiContexts,
    mut gen_params: ResMut<GenParams>,
    mut visual: ResMut<VisualParams>,
    mut regen: ResMut<RegenRequested>,
    last_regen: Res<LastRegenSecs>,
    stats: Option<Res<TerrainStats>>,
    capture: Option<Res<CaptureMode>>,
) -> Result {
    // Verification captures must show pixels from the reference viewpoints
    // only — no UI chrome.
    if capture.is_some() {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    egui::SidePanel::left("terrain-controls")
        .default_width(270.0)
        .show(ctx, |ui| {
            ui.heading("Terrain");

            ui.collapsing("Generation (press Regenerate)", |ui| {
                // bypass_change_detection: sliders write through ResMut every
                // frame they're touched; only the Regenerate button should
                // trigger the expensive path.
                let p = gen_params.bypass_change_detection();
                ui.add(egui::Slider::new(&mut p.seed, 0..=9999).text("seed"));
                ui.horizontal(|ui| {
                    ui.label("grid");
                    for g in [512usize, 1024, 2048] {
                        ui.selectable_value(&mut p.grid, g, g.to_string());
                    }
                });
                ui.add(egui::Slider::new(&mut p.amplitude, 2.0..=25.0).text("fBm amplitude"));
                ui.add(egui::Slider::new(&mut p.uplift_strength, 0.0..=30.0).text("uplift"));
                ui.add(egui::Slider::new(&mut p.ridge_strength, 0.0..=30.0).text("ridges"));
                ui.add(
                    egui::Slider::new(&mut p.talus_angle_deg, 25.0..=45.0).text("talus angle °"),
                );
                if ui.button("Regenerate").clicked() {
                    regen.0 = true;
                }
                if last_regen.0 > 0.0 {
                    ui.label(format!("last run: {:.2}s", last_regen.0));
                }
            });

            ui.collapsing("Environment (live)", |ui| {
                let v = visual.bypass_change_detection();
                let mut changed = false;
                changed |= ui
                    .add(egui::Slider::new(&mut v.sea_level_frac, 0.0..=0.4).text("sea level"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut v.snow_line, 0.3..=1.0).text("snow line"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut v.rock_slope_deg, 15.0..=60.0).text("rock slope °"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut v.grass_dirt_line, 0.0..=1.0).text("grass→dirt"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut v.sun_azimuth_deg, 0.0..=360.0).text("sun azimuth"))
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut v.sun_elevation_deg, 5.0..=89.0)
                            .text("sun elevation"),
                    )
                    .changed();
                if changed {
                    visual.set_changed();
                }
            });

            if let Some(stats) = stats {
                ui.separator();
                ui.label(format!(
                    "heights [{:.1}, {:.1}], range {:.1}",
                    stats.h_min, stats.h_max,
                    stats.h_max - stats.h_min
                ));
            }
        });
    Ok(())
}

/// Instant-apply path: water height, sun direction, biome thresholds. Runs
/// only when a panel slider actually changed the resource.
fn apply_visual_params(
    visual: Res<VisualParams>,
    stats: Option<Res<TerrainStats>>,
    handles: Option<Res<TerrainHandles>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut water: Query<&mut Transform, (With<WaterPlane>, Without<Sun>)>,
    mut sun: Query<&mut Transform, (With<Sun>, Without<WaterPlane>)>,
) {
    if !visual.is_changed() || visual.is_added() {
        return;
    }
    let (Some(stats), Some(handles)) = (stats, handles) else { return };

    if let Ok(mut t) = water.single_mut() {
        t.translation.y = stats.water_level(visual.sea_level_frac);
    }
    if let Ok(mut t) = sun.single_mut() {
        *t = sun_transform(visual.sun_azimuth_deg, visual.sun_elevation_deg);
    }
    if let Some(mat) = materials.get_mut(&handles.material) {
        mat.extension.params.snow_line = visual.snow_line;
        mat.extension.params.rock_slope_cos = visual.rock_slope_deg.to_radians().cos();
        mat.extension.params.grass_dirt_line = visual.grass_dirt_line;
    }
}

/// Expensive path: full pipeline re-run when the panel requested it. Replaces
/// the mesh asset in place and refreshes the material's height range — the
/// terrain entity itself never respawns. Blocks the frame for the duration
/// (~0.3 s at 512², ~4 s at 2048²).
fn regenerate_terrain(
    mut regen: ResMut<RegenRequested>,
    gen_params: Res<GenParams>,
    visual: Res<VisualParams>,
    gpu: Option<Res<GpuCompute>>,
    handles: Option<Res<TerrainHandles>>,
    mut stats_res: Option<ResMut<TerrainStats>>,
    mut last_regen: ResMut<LastRegenSecs>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut water: Query<&mut Transform, With<WaterPlane>>,
) {
    if !regen.0 {
        return;
    }
    regen.0 = false;
    let (Some(gpu), Some(handles), Some(stats_res)) = (gpu, handles, stats_res.as_deref_mut())
    else {
        return;
    };

    let t = std::time::Instant::now();
    let (hm, stats) = run_pipeline(&gpu.0, &gen_params);
    let mesh = heightmap_to_mesh(&hm, WORLD_SIZE, HEIGHT_SCALE);
    meshes
        .insert(&handles.mesh, mesh)
        .expect("terrain mesh handle is owned by TerrainHandles and never dropped");
    *stats_res = stats;

    if let Some(mat) = materials.get_mut(&handles.material) {
        mat.extension.params.world_y_min = stats.h_min;
        mat.extension.params.world_y_range = stats.range();
    }
    if let Ok(mut t) = water.single_mut() {
        t.translation.y = stats.water_level(visual.sea_level_frac);
    }
    last_regen.0 = t.elapsed().as_secs_f32();
    info!("regenerated in {:.2}s", last_regen.0);
}
