# Phase 8: Live Controls — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** egui panel that edits the terrain parameters live — completing build-order step 8 ("egui sliders for every TerrainParams field"). Two latency classes, two UX paths:
- **Generation params** (seed, grid size, fBm amplitude, uplift/ridge strength, talus angle) require a full pipeline re-run (~0.3 s at 512², ~4 s at 2048²) → applied on a **Regenerate** button.
- **Visual params** (sea level, snow line, rock slope angle, grass↔dirt line, sun azimuth/elevation) only touch a material uniform or a transform → applied **instantly** as the slider drags.

**Tech:** `bevy_egui 0.39.1` (pairs with Bevy 0.18.1; UI systems live in the `EguiPrimaryContextPass` schedule, `EguiPlugin::default()`). `bevy_panorbit_camera` gets its `bevy_egui` feature so slider drags don't orbit the camera.

## Architecture
- `GenParams` / `VisualParams` resources; `TerrainStats { h_min, h_max }`; `TerrainHandles { mesh, material }`; `GpuCompute(GpuContext)` created once; `RegenRequested(bool)`.
- `run_pipeline(&GpuContext, &GenParams) -> (Heightmap, min, max)` — the existing fbm→tectonics→hydraulic→thermal chain, parameterised. Erosion iterations stay derived from grid (`grid * 3/4`).
- Regeneration replaces the mesh asset in place (`meshes.insert(&handle, new_mesh)`) and updates the material's `world_y_min/range` — same entity, no respawn.
- Visual-apply system mutates: water plane `Transform.y`, sun `Transform` (spherical az/el), biome uniform via `Assets<TerrainMaterial>::get_mut`.

## Tasks
- [ ] Task 1: deps (`bevy_egui`, panorbit `bevy_egui` feature) — done during planning.
- [ ] Task 2: refactor `main.rs` — params resources, `run_pipeline`, markers (`WaterPlane`, `Sun`), handles resource.
- [ ] Task 3: `ui_panel` system in `EguiPrimaryContextPass` + `apply_visual_params` + `regenerate_terrain` systems.
- [ ] Task 4: verify — capture both viewpoints unchanged at defaults; `cargo nextest run` green; manual slider check in live preview; commit.

## Definition of done
- Panel shows generation + visual sections with sensible ranges; Regenerate rebuilds terrain with new seed/params; visual sliders update water/sun/biomes with zero lag.
- Defaults produce the same image as Phase 7 (captured from both reference viewpoints).
- All tests green.
