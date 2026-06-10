# TerraForge — Realistic Terrain Generator

**Date:** 2026-06-10
**Status:** Design approved (Milestone 1)
**Location:** `C:\Users\user\Documents\TerraForge\`

## Goal

A real-time, fly-around 3D terrain generator that produces the most realistic
terrain achievable with a procedural + physical-simulation pipeline. The user
can tweak parameters live and watch believable, geologically-plausible terrain
form on screen.

This document specs **Milestone 1: the realism core**. Later capabilities
(rivers, vegetation, climate, AI diffusion base, export) are explicitly deferred
and will get their own spec → plan cycles.

## Why this approach

Perceptual research and production tools (Gaea, World Machine, Houdini
HeightFields) converge on the same conclusion: realistic terrain is **not** pure
fractal noise. It is a *hybrid pipeline* — a procedural base shaped by
large-scale geological control, then refined by physical erosion simulation:

```
fBm/Perlin base
  → tectonic control (uplift, ridge guidance, slope constraints)
  → hydraulic erosion (GPU)      ← single biggest realism boost
  → thermal erosion (GPU)        ← talus slopes / angle of repose
  → biome texturing (slope + altitude)
  → water plane + sky/atmosphere
```

AI diffusion models (e.g. Terrain Diffusion Network, TerraFusion) are
state-of-the-art for generating the *base* layer from real DEM data, but they
require large datasets, heavy training, and still rely on physical erosion
behind them for hydrological coherence. They are deferred to a later milestone
and will plug in *in front of* the erosion stages without changing them.

## Stack

Chosen for maximum power **and** maximum reuse of existing wheels:

| Concern            | Choice                                  |
|--------------------|-----------------------------------------|
| Language           | Rust (edition 2021/2024)                |
| Engine / renderer  | **Bevy** (PBR, ECS, windowing, lighting)|
| GPU backend        | wgpu → Vulkan / DX12 / Metal            |
| Erosion simulation | wgpu **compute shaders (WGSL)** on GPU  |
| Camera             | `bevy_panorbit_camera` (fly-around)     |
| Live UI / sliders  | `bevy_egui` + egui                      |
| Procedural noise   | `noise` crate (CPU base) / WGSL noise   |
| Sky / atmosphere   | `bevy_atmosphere`                       |

Rationale for Rust + Bevy over C++: Bevy provides the renderer, lighting,
camera, and windowing out of the box and the crate ecosystem snaps together
(`cargo add ...`), so effort goes into the terrain pipeline rather than Vulkan
boilerplate. GPU performance is identical (same wgpu → Vulkan path). Rust's
compiler catches the buffer/aliasing bugs that make from-scratch GPU simulation
code painful in C++.

## Architecture

Clean, isolated units. The simulation stages share one contract —
**heightmap in → heightmap out** — so they compose, can be reordered, tested
independently, and the future AI base layer slots in without touching erosion.

```
src/
  main.rs            // Bevy app: plugin wiring, startup, run loop
  heightmap.rs       // Heightmap type: GPU texture + CPU mirror, dimensions
  noise/             // fBm / Perlin base generation -> Heightmap
  tectonics/         // uplift & ridge control applied to a Heightmap
  erosion/
    hydraulic.wgsl   // GPU compute: water flow, sediment transport/deposition
    thermal.wgsl     // GPU compute: talus / angle-of-repose smoothing
    mod.rs           // dispatch, buffers, CPU<->GPU sync
  render/
    mesh.rs          // Heightmap -> Bevy mesh (positions, normals, UVs)
    material.rs      // slope/altitude biome blending (rock/grass/snow/sand)
    water.rs         // sea-level water plane + shader
  ui/
    panel.rs         // egui parameter panel, drives regeneration
  params.rs          // central TerrainParams resource (all tunables)
```

### Data flow

1. `TerrainParams` (Bevy resource) holds every tunable (seed, octaves,
   frequency, uplift strength, erosion iterations, rain rate, sea level, ...).
2. On change (or startup), the pipeline runs: noise → tectonics → upload to GPU
   → N iterations hydraulic + thermal erosion (compute) → read back / keep on
   GPU → rebuild mesh + recompute normals → material reflects new slopes.
3. Render: PBR mesh with biome material, water plane at sea level, atmosphere.
4. egui panel edits `TerrainParams`; edits trigger regeneration (debounced).

### Heightmap contract

A single `Heightmap` abstraction wraps an `R32Float` GPU texture (plus an
optional CPU `Vec<f32>` mirror) of size `N×N` (default 512, configurable to
1024/2048). Every generation/erosion stage consumes and produces this type.
This is the seam the AI diffusion layer will later sit behind.

## Error handling

- GPU adapter / device / shader-compile failures: surface a clear message and
  exit cleanly (no silent fallback to a broken state).
- Heightmap dimension mismatches between stages: debug-assert + typed
  dimensions to make them unrepresentable in normal flow.
- Erosion divergence (NaN/Inf from bad parameters): clamp heights per-iteration
  and validate params at the UI boundary (ranges on sliders).
- No fallback that hides failure — if a stage can't run, say so.

## Testing

- **Unit (CPU):** noise determinism (same seed → same heightmap), tectonic
  uplift monotonicity, mesh normal correctness on known heightfields,
  param-range validation.
- **GPU/integration:** run erosion N iterations on a known input; assert
  invariants — total mass roughly conserved (within sediment tolerance),
  no NaN/Inf, peaks lowered & valleys filled (drainage forms), output stays in
  bounds. Use a small headless heightmap so it runs in CI.
- **Visual/manual:** the viewer itself is the acceptance test for realism;
  capture before/after-erosion screenshots for the spec record.

## Build order (foundations first)

1. **Foundation** — Bevy app + window + fly camera + lighting; render a mesh
   from a trivial heightmap. *Get pixels on screen.*
2. **Procedural base** — multi-octave fBm → believable large-scale shape.
3. **Tectonic control** — uplift / ridge guidance for structured mountains.
4. **GPU hydraulic erosion** — the core realism boost.
5. **GPU thermal erosion** — natural talus slopes.
6. **Biome texturing** — rock/grass/snow/sand by slope + altitude.
7. **Water + sky** — sea-level water plane, atmosphere.
8. **Live controls** — egui sliders for every `TerrainParams` field.

## Out of scope (future milestones, separate specs)

- River-network mesh solving / hydrological graph
- Vegetation scattering (trees, ground cover)
- Climate / precipitation simulation driving biomes
- **AI diffusion base layer** trained on real DEMs (SRTM / Copernicus)
- Heightmap / mesh export for external engines (UE5, Unity, Blender)
- Endless / chunked / planet-scale terrain

## Success criteria (Milestone 1)

- A fly-around window shows terrain produced by the full hybrid pipeline.
- Hydraulic + thermal erosion run on the GPU and visibly carve valleys / form
  talus, toggleable and tunable live.
- Biome texturing blends by slope and altitude; a water plane sits at an
  adjustable sea level under an atmospheric sky.
- Every key parameter is a live egui slider; changing it regenerates terrain.
- Determinism: a given seed + params reproduces the same terrain.
