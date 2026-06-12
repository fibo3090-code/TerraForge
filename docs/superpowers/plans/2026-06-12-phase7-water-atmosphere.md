# Phase 7: Water Plane + Atmosphere — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the flat sky-blue clear colour with Bevy 0.18's physically-based atmospheric scattering, and put a translucent water plane at a sea level derived from the post-erosion heightmap stats — completing build-order step 7 ("water + sky").

**Architecture:** All camera-side: `Atmosphere::earthlike(ScatteringMedium::default())` + `AtmosphereSettings { scene_units_to_m: 100.0 }` (1 world unit = 100 m → the 100-unit terrain reads as a 10 km massif and far ridges pick up aerial haze) + `Exposure { ev100: 13.0 }` + `Bloom::NATURAL`. Sun raised to `lux::RAW_SUNLIGHT` so the atmosphere LUTs are driven at physical intensity; `GlobalAmbientLight::NONE` + `AtmosphereEnvironmentMapLight` so ambient comes from the sky itself. Water is a `Plane3d` at `y = h_min + 0.12 * range` with an alpha-blended low-roughness `StandardMaterial`.

**Bevy 0.18 specifics (verified against registry source + `examples/3d/atmosphere.rs`):**
- `Atmosphere` `#[require(AtmosphereSettings, Hdr)]` — Hdr comes in automatically.
- Needs `Assets<ScatteringMedium>` in the spawn system.
- Example pairs atmosphere with `Msaa::Off` + `Fxaa` (render-sky node and MSAA interact badly); follow that.
- Imports: `bevy::camera::Exposure`, `bevy::post_process::bloom::Bloom`, `bevy::anti_alias::fxaa::Fxaa`, `bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight}`, `bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium}`.

## Task 1: Atmosphere on the camera
- [ ] Remove `ClearColor` sky-blue + `GlobalAmbientLight` resources (atmosphere replaces both; ambient → `GlobalAmbientLight::NONE`).
- [ ] Camera gains `Atmosphere::earthlike(...)`, `AtmosphereSettings { scene_units_to_m: 100.0, ..default() }`, `Exposure { ev100: 13.0 }`, `Bloom::NATURAL`, `AtmosphereEnvironmentMapLight::default()`, `Msaa::Off`, `Fxaa::default()`. Tonemapping → `AcesFitted` (example-matched).
- [ ] Sun illuminance → `lux::RAW_SUNLIGHT`; lower elevation angle for warmer raking light.

## Task 2: Water plane
- [ ] `Plane3d` at 4× WORLD_SIZE, `y = h_min + 0.12 * (h_max - h_min)`, alpha-blend StandardMaterial (deep blue-green, roughness 0.08), `NotShadowCaster`.

## Task 3: Verify
- [ ] `cargo nextest run` all green (pipeline untouched, expect no change).
- [ ] `TERRAFORGE_CAPTURE` close-up + wide captures: sky gradient visible (not flat blue), water visible in low valleys, no regression in terrain texturing (same-viewpoint check vs `test_output/20_repeat_mode.png`).
- [ ] Commit.

## Definition of done
- Sky is a real scattering gradient with sun disc bloom; far terrain shows aerial perspective.
- Water plane floods the lowest valleys at a level derived from this seed's stats.
- All tests green; captured renders verified from both reference viewpoints.
