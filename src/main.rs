#![allow(clippy::too_many_arguments)] // Bevy systems take their deps as params

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use bevy::anti_alias::fxaa::Fxaa;
use bevy::camera::Exposure;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{
    light_consts::lux, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder,
    DirectionalLightShadowMap, GlobalAmbientLight, NotShadowCaster,
};
use bevy::pbr::{Atmosphere, AtmosphereSettings, MaterialPlugin, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use bevy::tasks::futures_lite::future;
use bevy::tasks::{block_on, AsyncComputeTaskPool, Task};
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

mod analysis;
mod biome;
mod erosion;
mod export;
mod heightmap;
mod hydrology;
mod mesh_builder;
mod pipeline;
mod talus;
mod tectonics;
mod terrain_noise;

use biome::{build_terrain_material, BiomeParams, TerrainMaterial};
use heightmap::Heightmap;
use hydrology::WaterField;
use mesh_builder::heightmap_to_mesh;
use pipeline::{progress, PipelineParams, PipelineRun, StageCache, METERS_PER_UNIT};

const HEIGHT_SCALE: f32 = 1.0;

// ---------------------------------------------------------------------------
// Parameters (edited live from the egui panel)
// ---------------------------------------------------------------------------

/// Parameters that apply instantly (uniform / transform mutation only).
#[derive(Resource, Clone, PartialEq)]
struct VisualParams {
    /// Sea level as a fraction of the heightmap range above its minimum.
    sea_level_frac: f32,
    snow_line: f32,
    rock_slope_deg: f32,
    grass_dirt_line: f32,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    water_color: [f32; 3],
    water_roughness: f32,
}

impl Default for VisualParams {
    fn default() -> Self {
        Self {
            sea_level_frac: 0.12,
            snow_line: 0.62,
            rock_slope_deg: 32.0,
            grass_dirt_line: 0.45,
            sun_azimuth_deg: 33.7,
            sun_elevation_deg: 59.0,
            water_color: [0.08, 0.22, 0.32],
            water_roughness: 0.08,
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

/// The headless compute context survives regenerations and is shared with
/// background regen tasks.
#[derive(Resource)]
struct GpuCompute(Arc<erosion::GpuContext>);

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

/// Asset handles that live across regenerations: meshes are replaced in
/// place, material uniforms are mutated.
#[derive(Resource)]
struct TerrainHandles {
    mesh: Handle<Mesh>,
    material: Handle<TerrainMaterial>,
    water_mesh: Handle<Mesh>,
    water_material: Handle<StandardMaterial>,
    /// Lake/river water-surface mesh (rebuilt every regen).
    river_mesh: Handle<Mesh>,
}

#[derive(Resource, Default)]
struct RegenRequested(bool);

/// Stage cache, taken by the background task during a regen and restored on
/// completion. `None` while a regen is in flight.
#[derive(Resource, Default)]
struct PipelineCache(Option<StageCache>);

struct RegenOutput {
    run: PipelineRun,
    mesh: Mesh,
    cache: StageCache,
    total_secs: f32,
}

#[derive(Resource, Default)]
struct RegenInFlight {
    task: Option<Task<RegenOutput>>,
    progress: Arc<AtomicU8>,
}

/// Panel feedback: which stages ran/reused last time + timings.
#[derive(Resource, Default)]
struct LastRunInfo(String);

/// Post-regen invariant audit result for the panel.
#[derive(Resource, Default)]
struct LastAudit(Option<Result<(), String>>);

/// The live post-pipeline heightmap (export source).
#[derive(Resource)]
struct CurrentHeightmap(Heightmap);

/// The live water field (lake/river surface + depth) for export.
#[derive(Resource)]
struct CurrentWater(WaterField);

#[derive(Resource, Default)]
struct ExportInFlight {
    task: Option<Task<Result<String, String>>>,
    last_result: Option<Result<String, String>>,
}

/// Accessibility / UI preferences.
#[derive(Resource)]
struct UiPrefs {
    /// egui pixels-per-point multiplier (text + widget size).
    scale: f32,
    /// F1 toggles the whole panel for an unobstructed view.
    visible: bool,
}

impl Default for UiPrefs {
    fn default() -> Self {
        Self { scale: 1.0, visible: true }
    }
}

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
        .insert_resource(PipelineParamsRes::default())
        .insert_resource(VisualParams::default())
        .insert_resource(RegenRequested::default())
        .insert_resource(PipelineCache::default())
        .insert_resource(RegenInFlight::default())
        .insert_resource(LastRunInfo::default())
        .insert_resource(LastAudit::default())
        .insert_resource(ExportInFlight::default())
        .insert_resource(UiPrefs::default())
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(EguiPlugin::default())
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ui_panel)
        .add_systems(Update, (apply_visual_params, spawn_regen, poll_regen, poll_export));

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

/// Newtype so the egui panel can hold `PipelineParams` as a Bevy resource.
#[derive(Resource, Default)]
struct PipelineParamsRes(PipelineParams);

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
// Setup
// ---------------------------------------------------------------------------

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    params: Res<PipelineParamsRes>,
    visual: Res<VisualParams>,
    mut regen: ResMut<RegenRequested>,
    mut cache: ResMut<PipelineCache>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let gpu = Arc::new(
        erosion::GpuContext::new()
            .expect("GPU compute unavailable - hydraulic erosion requires a wgpu adapter"),
    );
    let mut stage_cache = StageCache::default();
    let progress = AtomicU8::new(0);
    // Fast first frame: build a 512^2 preview synchronously (~0.4 s) and
    // queue the configured-resolution build on the async path right away.
    // Capture mode keeps the synchronous full-res build so the verification
    // reference pixels are unaffected.
    let mut startup_params = params.0.clone();
    if std::env::var("TERRAFORGE_CAPTURE").is_err() && startup_params.base.grid > 512 {
        startup_params.base.grid = 512;
        regen.0 = true;
    }
    let run = pipeline::run_pipeline(&gpu, &startup_params, &mut stage_cache, &progress);
    cache.0 = Some(stage_cache);
    commands.insert_resource(GpuCompute(gpu));
    let stats = TerrainStats { h_min: run.h_min, h_max: run.h_max };
    commands.insert_resource(stats);

    let world_size = params.0.base.world_size;
    let mesh = heightmap_to_mesh(&run.heightmap, world_size, HEIGHT_SCALE);
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
    commands.spawn((Mesh3d(mesh_handle.clone()), MeshMaterial3d(material_handle.clone())));

    // Water plane: floods the lowest valleys; sea level derives from stats.
    let water_level = stats.water_level(visual.sea_level_frac);
    let water_mesh_handle = meshes.add(water_plane_mesh(world_size));
    let water_material_handle = std_materials.add(StandardMaterial {
        base_color: water_color(&visual),
        perceptual_roughness: visual.water_roughness,
        metallic: 0.0,
        reflectance: 0.4,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    commands.spawn((
        Mesh3d(water_mesh_handle.clone()),
        MeshMaterial3d(water_material_handle.clone()),
        Transform::from_xyz(0.0, water_level, 0.0),
        NotShadowCaster,
        WaterPlane,
    ));
    // Lake/river surface mesh from the hydrology water field.
    let river_mesh_handle =
        meshes.add(water_surface_mesh(&run.heightmap, &run.water, world_size));
    commands.spawn((
        Mesh3d(river_mesh_handle.clone()),
        MeshMaterial3d(water_material_handle.clone()),
        NotShadowCaster,
    ));
    commands.insert_resource(CurrentWater(run.water.clone()));
    commands.insert_resource(CurrentHeightmap(run.heightmap));
    commands.insert_resource(TerrainHandles {
        mesh: mesh_handle,
        material: material_handle,
        water_mesh: water_mesh_handle,
        water_material: water_material_handle,
        river_mesh: river_mesh_handle,
    });

    // Sun at physical illuminance — the atmosphere LUTs expect real-world
    // lux; the camera's Exposure brings it back to display range.
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        sun_transform(visual.sun_azimuth_deg, visual.sun_elevation_deg, world_size),
        shadow_config(world_size),
        Sun,
    ));

    // Capture viewpoints: "close" (default) frames a slope where texture
    // artifacts are loudest; "wide" frames the whole terrain + horizon.
    // Always capture BOTH when validating a visual change.
    let scale = world_size / 100.0;
    let view = std::env::var("TERRAFORGE_CAPTURE_VIEW").unwrap_or_default();
    let (cam_pos, cam_target) = if std::env::var("TERRAFORGE_CAPTURE").is_ok() {
        if view == "wide" {
            (Vec3::new(0.0, 55.0, 150.0) * scale, Vec3::new(0.0, 5.0 * scale, 0.0))
        } else {
            let mid = (stats.h_min + stats.h_max) * 0.5;
            (
                Vec3::new(15.0 * scale, stats.h_max + 8.0, 15.0 * scale),
                Vec3::new(-25.0 * scale, mid, -25.0 * scale),
            )
        }
    } else {
        (Vec3::new(0.0, 0.7 * world_size, 1.3 * world_size), Vec3::ZERO)
    };
    commands.spawn((
        Camera3d::default(),
        // Physically-based sky + aerial perspective. scene_units_to_m = 20
        // reads the 100-unit terrain as a ~2 km massif: subtle haze on far
        // ridges without fogging the scene.
        Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
        AtmosphereSettings { scene_units_to_m: METERS_PER_UNIT, ..default() },
        AtmosphereEnvironmentMapLight::default(),
        Exposure { ev100: 13.0 },
        Bloom::NATURAL,
        // The atmosphere render-sky node is incompatible with MSAA; pair
        // with FXAA per the upstream example.
        Msaa::Off,
        Fxaa::default(),
        Tonemapping::AcesFitted,
        Transform::from_translation(cam_pos).looking_at(cam_target, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}

fn water_plane_mesh(world_size: f32) -> Mesh {
    // 20x the terrain so the plane's edge stays beyond the horizon.
    Plane3d::default().mesh().size(world_size * 20.0, world_size * 20.0).into()
}

/// Lake/river surface mesh: wet cells sit at the water-surface elevation,
/// dry cells are tucked just below the terrain so the surface hugs the
/// shoreline. Downsampled to <=1024 per side — water is smooth.
fn water_surface_mesh(terrain: &Heightmap, water: &WaterField, world_size: f32) -> Mesh {
    let stride = (terrain.width / 1024).max(1);
    let w = (terrain.width / stride).max(2);
    let h = (terrain.height / stride).max(2);
    let hm = Heightmap::from_fn(w, h, |x, z| {
        let sx = (x * stride).min(terrain.width - 1);
        let sz = (z * stride).min(terrain.height - 1);
        let s = water.surface.get(sx, sz);
        if s.is_finite() {
            s
        } else {
            terrain.get(sx, sz) - 0.3
        }
    });
    heightmap_to_mesh(&hm, world_size, HEIGHT_SCALE)
}

fn water_color(v: &VisualParams) -> Color {
    Color::srgba(v.water_color[0], v.water_color[1], v.water_color[2], 0.9)
}

fn sun_transform(azimuth_deg: f32, elevation_deg: f32, world_size: f32) -> Transform {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.to_radians();
    let dir = Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos());
    Transform::from_translation(dir * 2.0 * world_size).looking_at(Vec3::ZERO, Vec3::Y)
}

fn shadow_config(world_size: f32) -> bevy::light::CascadeShadowConfig {
    CascadeShadowConfigBuilder {
        num_cascades: 4,
        first_cascade_far_bound: 0.3 * world_size,
        maximum_distance: 4.0 * world_size,
        ..default()
    }
    .build()
}

// ---------------------------------------------------------------------------
// Live controls
// ---------------------------------------------------------------------------

fn ui_panel(
    mut contexts: EguiContexts,
    mut params: ResMut<PipelineParamsRes>,
    mut visual: ResMut<VisualParams>,
    mut regen: ResMut<RegenRequested>,
    regen_inflight: Res<RegenInFlight>,
    mut export_inflight: ResMut<ExportInFlight>,
    current_hm: Option<Res<CurrentHeightmap>>,
    current_water: Option<Res<CurrentWater>>,
    last_run: Res<LastRunInfo>,
    last_audit: Res<LastAudit>,
    stats: Option<Res<TerrainStats>>,
    capture: Option<Res<CaptureMode>>,
    mut prefs: ResMut<UiPrefs>,
    diagnostics: Res<DiagnosticsStore>,
) -> Result {
    // Verification captures must show pixels from the reference viewpoints
    // only — no UI chrome.
    if capture.is_some() {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    ctx.set_pixels_per_point(prefs.scale);
    // Keyboard shortcuts (suppressed while a text field has focus).
    let typing = ctx.wants_keyboard_input();
    if !typing && ctx.input(|i| i.key_pressed(egui::Key::F1)) {
        prefs.visible = !prefs.visible;
    }
    if !typing
        && ctx.input(|i| i.key_pressed(egui::Key::R))
        && regen_inflight.task.is_none()
    {
        regen.0 = true;
    }
    if !prefs.visible {
        return Ok(());
    }
    egui::SidePanel::left("terrain-controls")
        .default_width(290.0)
        .show(ctx, |ui| {
            ui.heading("Terrain");
            let busy = regen_inflight.task.is_some();

            ui.collapsing("Generation (press Regenerate)", |ui| {
                let p = &mut params.bypass_change_detection().0;
                ui.add(egui::Slider::new(&mut p.base.seed, 0..=99_999).text("seed"))
                    .on_hover_text("Same seed + same params = bit-identical terrain");
                ui.horizontal(|ui| {
                    if ui.button("🎲 random seed").clicked() {
                        p.base.seed = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .subsec_nanos()
                            % 100_000;
                        regen.0 = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("grid");
                    for g in [512usize, 1024, 2048] {
                        ui.selectable_value(&mut p.base.grid, g, g.to_string());
                    }
                });
                ui.add(
                    egui::Slider::new(&mut p.base.world_size, 50.0..=500.0)
                        .text("map size")
                        .custom_formatter(|v, _| {
                            format!("{:.1} km", v as f32 * METERS_PER_UNIT / 1000.0)
                        }),
                );
                ui.label(format!(
                    "cell size: {:.1} m",
                    p.base.world_size / p.base.grid as f32 * METERS_PER_UNIT
                ))
                .on_hover_text("Map size / grid — finer cells hold finer erosion detail");
                ui.separator();
                ui.add(egui::Slider::new(&mut p.base.amplitude, 2.0..=25.0).text("fBm amplitude"))
                    .on_hover_text("Base noise relief in world units (1 unit = 20 m)");
                ui.add(egui::Slider::new(&mut p.base.octaves, 1..=10).text("fBm octaves"));
                ui.add(egui::Slider::new(&mut p.base.frequency, 0.5..=8.0).text("fBm frequency"));
                ui.add(
                    egui::Slider::new(&mut p.base.persistence, 0.2..=0.8).text("fBm persistence"),
                );
                ui.separator();
                ui.add(
                    egui::Slider::new(&mut p.tectonics.uplift_strength, 0.0..=30.0).text("uplift"),
                );
                ui.add(
                    egui::Slider::new(&mut p.tectonics.ridge_strength, 0.0..=30.0).text("ridges"),
                );
                ui.add(
                    egui::Slider::new(&mut p.tectonics.uplift_frequency, 0.2..=4.0)
                        .text("uplift frequency"),
                );
                ui.add(
                    egui::Slider::new(&mut p.tectonics.ridge_frequency, 0.5..=8.0)
                        .text("ridge frequency"),
                );
                ui.separator();
                ui.add(
                    egui::Slider::new(&mut p.hydraulic.rain_rate, 0.002..=0.05)
                        .logarithmic(true)
                        .text("rain rate"),
                )
                .on_hover_text("Water per erosion step — more rain cuts deeper valleys");
                ui.add(
                    egui::Slider::new(&mut p.hydraulic.capacity_k, 0.2..=2.0)
                        .text("sediment capacity"),
                );
                ui.add(
                    egui::Slider::new(&mut p.hydraulic.total_dig_budget, 0.05..=0.5)
                        .text("dig budget"),
                );
                ui.add(
                    egui::Slider::new(&mut p.thermal.talus_angle_deg, 25.0..=45.0)
                        .text("talus angle °"),
                );
                ui.separator();
                ui.add(
                    egui::Slider::new(&mut p.hydrology.river_threshold, 0.0002..=0.01)
                        .logarithmic(true)
                        .text("river density"),
                )
                .on_hover_text("Low = dense river network, high = only the major rivers");
                ui.add(
                    egui::Slider::new(&mut p.hydrology.carve_depth, 0.0..=1.0)
                        .text("river carve depth"),
                );
                ui.add(
                    egui::Slider::new(&mut p.hydrology.river_width, 0.0..=6.0)
                        .text("river width"),
                );
                ui.separator();
                if busy {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        let stage = regen_inflight.progress.load(Ordering::Relaxed);
                        ui.label(format!("running: {}", progress::label(stage)));
                    });
                } else if ui.button("Regenerate").clicked() {
                    regen.0 = true;
                }
                if !last_run.0.is_empty() {
                    ui.label(&last_run.0);
                }
                match &last_audit.0 {
                    Some(Ok(())) => {
                        ui.colored_label(egui::Color32::from_rgb(80, 200, 80), "✓ invariants ok");
                    }
                    Some(Err(msg)) => {
                        ui.colored_label(egui::Color32::from_rgb(230, 80, 80), msg);
                    }
                    None => {}
                }
            });

            ui.collapsing("Environment (live)", |ui| {
                let v = visual.bypass_change_detection();
                let mut changed = false;
                changed |= ui
                    .add(egui::Slider::new(&mut v.sea_level_frac, 0.0..=0.4).text("sea level"))
                    .on_hover_text("Ocean height as a fraction of the terrain's relief")
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut v.snow_line, 0.3..=1.0).text("snow line"))
                    .on_hover_text("Normalized elevation where snow appears on gentle slopes")
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut v.rock_slope_deg, 15.0..=60.0).text("rock slope °"),
                    )
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut v.grass_dirt_line, 0.0..=1.0).text("grass→dirt"))
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut v.sun_azimuth_deg, 0.0..=360.0).text("sun azimuth"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut v.sun_elevation_deg, 5.0..=89.0)
                            .text("sun elevation"),
                    )
                    .changed();
                ui.horizontal(|ui| {
                    ui.label("water colour");
                    changed |= ui.color_edit_button_rgb(&mut v.water_color).changed();
                });
                changed |= ui
                    .add(
                        egui::Slider::new(&mut v.water_roughness, 0.02..=0.5)
                            .text("water roughness"),
                    )
                    .changed();
                if changed {
                    visual.set_changed();
                }
            });

            ui.collapsing("Export", |ui| {
                if export_inflight.task.is_some() {
                    ui.spinner();
                } else if ui.button("Export heightmap + masks…").clicked() {
                    if let (Some(hm), false) = (current_hm.as_ref(), busy) {
                        if let Some(dir) =
                            rfd::FileDialog::new().set_title("Export folder").pick_folder()
                        {
                            let hm = hm.0.clone();
                            let water = current_water.as_ref().map(|w| w.0.clone());
                            let p = params.0.clone();
                            let frac = visual.sea_level_frac;
                            export_inflight.task =
                                Some(AsyncComputeTaskPool::get().spawn(async move {
                                    export::export_all(&hm, &p, frac, water.as_ref(), &dir)
                                        .map(|files| {
                                            format!(
                                                "exported {} files to {}",
                                                files.len(),
                                                dir.display()
                                            )
                                        })
                                }));
                        }
                    }
                }
                match &export_inflight.last_result {
                    Some(Ok(msg)) => {
                        ui.colored_label(egui::Color32::from_rgb(80, 200, 80), msg);
                    }
                    Some(Err(msg)) => {
                        ui.colored_label(egui::Color32::from_rgb(230, 80, 80), msg);
                    }
                    None => {}
                }
            });

            ui.separator();
            ui.add(egui::Slider::new(&mut prefs.scale, 0.75..=2.0).text("UI scale"))
                .on_hover_text("Accessibility: scales all panel text and widgets");
            if let Some(fps) = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
            {
                ui.label(format!("{fps:.0} fps"));
            }
            ui.label("R = regenerate · F1 = hide/show UI")
                .on_hover_text("Shortcuts work whenever no text field is focused");
            if let Some(stats) = stats {
                ui.separator();
                ui.label(format!(
                    "heights [{:.1}, {:.1}] · range {:.1} ({:.0} m)",
                    stats.h_min,
                    stats.h_max,
                    stats.range(),
                    stats.range() * METERS_PER_UNIT,
                ));
            }
        });
    Ok(())
}

/// Instant-apply path: water height/colour, sun direction, biome thresholds.
fn apply_visual_params(
    visual: Res<VisualParams>,
    params: Res<PipelineParamsRes>,
    stats: Option<Res<TerrainStats>>,
    handles: Option<Res<TerrainHandles>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
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
        *t = sun_transform(
            visual.sun_azimuth_deg,
            visual.sun_elevation_deg,
            params.0.base.world_size,
        );
    }
    if let Some(mat) = materials.get_mut(&handles.material) {
        mat.extension.params.snow_line = visual.snow_line;
        mat.extension.params.rock_slope_cos = visual.rock_slope_deg.to_radians().cos();
        mat.extension.params.grass_dirt_line = visual.grass_dirt_line;
    }
    if let Some(mat) = std_materials.get_mut(&handles.water_material) {
        mat.base_color = water_color(&visual);
        mat.perceptual_roughness = visual.water_roughness;
    }
}

// ---------------------------------------------------------------------------
// Async regeneration
// ---------------------------------------------------------------------------

fn spawn_regen(
    mut regen: ResMut<RegenRequested>,
    mut inflight: ResMut<RegenInFlight>,
    mut cache: ResMut<PipelineCache>,
    params: Res<PipelineParamsRes>,
    gpu: Option<Res<GpuCompute>>,
) {
    if !regen.0 || inflight.task.is_some() {
        return;
    }
    regen.0 = false;
    let Some(gpu) = gpu else { return };
    // Cache is unavailable only if a previous task panicked mid-flight; a
    // fresh (empty) cache is functionally identical, just slower.
    let mut stage_cache = cache.0.take().unwrap_or_default();
    let gpu = gpu.0.clone();
    let p = params.0.clone();
    let progress = inflight.progress.clone();
    inflight.task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let t = std::time::Instant::now();
        let run = pipeline::run_pipeline(&gpu, &p, &mut stage_cache, &progress);
        progress.store(progress::MESH, Ordering::Relaxed);
        let mesh = heightmap_to_mesh(&run.heightmap, p.base.world_size, HEIGHT_SCALE);
        progress.store(progress::DONE, Ordering::Relaxed);
        RegenOutput { run, mesh, cache: stage_cache, total_secs: t.elapsed().as_secs_f32() }
    }));
}

fn poll_regen(
    mut commands: Commands,
    mut inflight: ResMut<RegenInFlight>,
    mut cache: ResMut<PipelineCache>,
    mut last_run: ResMut<LastRunInfo>,
    mut last_audit: ResMut<LastAudit>,
    params: Res<PipelineParamsRes>,
    visual: Res<VisualParams>,
    handles: Option<Res<TerrainHandles>>,
    mut stats_res: Option<ResMut<TerrainStats>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut water: Query<&mut Transform, With<WaterPlane>>,
    sun: Query<Entity, With<Sun>>,
) {
    let Some(task) = inflight.task.as_mut() else { return };
    let Some(output) = block_on(future::poll_once(task)) else { return };
    inflight.task = None;
    let (Some(handles), Some(stats_res)) = (handles, stats_res.as_deref_mut()) else { return };

    let stats = TerrainStats { h_min: output.run.h_min, h_max: output.run.h_max };
    *stats_res = stats;

    meshes
        .insert(&handles.mesh, output.mesh)
        .expect("terrain mesh handle is owned by TerrainHandles and never dropped");
    // World size may have changed: refresh the dependent water mesh, sun
    // distance and shadow cascade extents.
    let world_size = params.0.base.world_size;
    meshes
        .insert(&handles.water_mesh, water_plane_mesh(world_size))
        .expect("water mesh handle is owned by TerrainHandles and never dropped");
    meshes
        .insert(
            &handles.river_mesh,
            water_surface_mesh(&output.run.heightmap, &output.run.water, world_size),
        )
        .expect("river mesh handle is owned by TerrainHandles and never dropped");
    if let Ok(e) = sun.single() {
        commands.entity(e).insert((
            sun_transform(visual.sun_azimuth_deg, visual.sun_elevation_deg, world_size),
            shadow_config(world_size),
        ));
    }

    if let Some(mat) = materials.get_mut(&handles.material) {
        mat.extension.params.world_y_min = stats.h_min;
        mat.extension.params.world_y_range = stats.range();
    }
    if let Ok(mut t) = water.single_mut() {
        t.translation.y = stats.water_level(visual.sea_level_frac);
    }

    // Panel feedback: reuse + timing summary.
    let reused = if output.run.reused.is_empty() {
        String::new()
    } else {
        format!("reused {} · ", output.run.reused.join(", "))
    };
    let ran: Vec<String> = output
        .run
        .stages_run
        .iter()
        .map(|(name, secs)| format!("{name} {secs:.2}s"))
        .collect();
    last_run.0 = format!(
        "{reused}{} · total {:.2}s · {} river / {} lake cells",
        ran.join(", "),
        output.total_secs,
        output.run.river_cells,
        output.run.lake_cells,
    );
    info!("regen: {}", last_run.0);

    // Objective audit: cheap invariants on every regen. Carved river
    // channels are intentional steps, so the spike threshold must clear the
    // carve depth.
    let cell_size = world_size / params.0.base.grid as f32;
    let max_drop = params.0.thermal.talus_angle_deg.to_radians().tan() * cell_size;
    let spike_threshold = (2.0 * max_drop).max(params.0.hydrology.carve_depth * 1.2);
    last_audit.0 = Some(audit_heightmap(
        &output.run.heightmap,
        output.run.tectonic_relief,
        spike_threshold,
    ));

    commands.insert_resource(CurrentWater(output.run.water.clone()));
    commands.insert_resource(CurrentHeightmap(output.run.heightmap));
    cache.0 = Some(output.cache);
}

fn poll_export(mut export_inflight: ResMut<ExportInFlight>) {
    let Some(task) = export_inflight.task.as_mut() else { return };
    let Some(result) = block_on(future::poll_once(task)) else { return };
    export_inflight.task = None;
    export_inflight.last_result = Some(result);
}

/// Post-regen invariant audit (objective check, shown in the panel).
fn audit_heightmap(hm: &Heightmap, tectonic_relief: f32, spike_threshold: f32) -> Result<(), String> {
    if !hm.data().iter().all(|v| v.is_finite()) {
        return Err("audit: non-finite heights".into());
    }
    let spikes = analysis::spike_mask(hm, spike_threshold).iter().filter(|&&b| b).count();
    let pct = spikes as f32 / hm.data().len() as f32;
    if pct > 0.005 {
        return Err(format!("audit: spike density {:.2}% > 0.5%", pct * 100.0));
    }
    let (mn, mx) = pipeline::min_max(hm);
    let relief = mx - mn;
    if relief < 0.5 * tectonic_relief || relief > 1.2 * tectonic_relief {
        return Err(format!(
            "audit: relief {relief:.1} outside [50%,120%] of tectonic {tectonic_relief:.1}"
        ));
    }
    Ok(())
}
