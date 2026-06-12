# TerraForge Milestone 2A: Workbench — Design Spec

**Date:** 2026-06-12
**Status:** Approved pending user review
**Context:** Milestone 1 (Phases 1–8) is complete: fBm → tectonics → GPU hydraulic erosion → GPU thermal erosion → biome texturing → water + atmosphere → live egui controls. Milestone 2 was decomposed into three sub-projects: **A — Workbench** (this spec), **B — Hydrology** (rivers/lakes), **C — Climate layer** (temperature, snow, glaciers, seasons, wind). A is built first because it makes B and C faster to develop and verify.

## Goal

Turn the app from "demo with sliders" into a usable terrain workbench:
1. Map size + full parameter exposure in the panel.
2. Async regeneration — the UI never freezes.
3. **Incremental pipeline** — only re-run the stages a parameter change actually affects.
4. Heightmap + mask export for game engines (Unreal/Unity) and DCC tools (Blender/Gaea/Houdini).
5. A verification protocol (objective + subjective) wired into the workflow.

## Non-goals

Rivers, lakes beyond the flat sea plane, climate simulation, vegetation — those are sub-projects B and C. No graph/node UI; the pipeline stays a fixed cascade.

## 1. Map size

- `world_size: f32` joins `GenParams` (range 50–500 world units; panel displays it as km at the fixed render scale of 20 m/unit, i.e. 1–10 km).
- **Noise and tectonic frequencies scale proportionally with `world_size / 100`** so feature density per km stays constant: a bigger map has *more* mountains, not stretched ones.
- Erosion cell size = `world_size / grid` (already parameterised); thermal `max_drop` derives from it (already does).
- Water plane extent, shadow `maximum_distance`, capture-camera positions, and the default orbit camera distance derive from `world_size` instead of constants.
- Panel shows derived cell size in metres next to the grid selector.

## 2. Expanded settings

Generation section additions (all require Regenerate):
- fBm: `octaves` (1–10), `frequency` (0.5–8), `persistence` (0.2–0.8).
- Tectonics: `uplift_frequency` (0.2–4), `ridge_frequency` (0.5–8) — strengths already exist.
- Hydraulic: `rain_rate` (0.002–0.05), `capacity_k` (0.2–2), `total_dig_budget` (0.05–0.5).
- `world_size` + existing seed/grid/amplitude/uplift/ridge/talus.
- A "🎲" button that randomizes the seed and immediately triggers a regen.

Environment section additions (instant): water `base_color` (color picker) and `perceptual_roughness` (0.02–0.5). Requires keeping the water material handle in a resource.

## 3. Async regeneration

**Approach: Bevy `AsyncComputeTaskPool`** (chosen over raw threads and frame-slicing). The full pipeline (cache-aware stage runs + `heightmap_to_mesh`) executes in a background task. `GpuContext` is headless and `wgpu` Device/Queue are `Send + Sync`; the context moves into an `Arc` shared between the task and the `GpuCompute` resource.

- `RegenTask` resource: `Option<Task<RegenOutput>>` + `Arc<AtomicU8>` progress stage (Base=0, Tectonics=1, Hydraulic=2, Thermal=3, Mesh=4, Done=5).
- `poll_regen` system: when the task completes, swap the mesh asset in place (same entity/material, as today), update `TerrainStats`, biome `world_y_min/range`, water level, cached stage outputs, and the post-regen audit (see §6).
- Panel: Regenerate button disabled while a task is in flight; stage label + spinner shown instead. One regen at a time; requests during flight are ignored (the button is disabled, so this is only a race guard).
- `RegenOutput`: `{ heightmap, mesh, stats, stage_cache_updates, timings }`.

## 4. Incremental pipeline (stage caching)

The pipeline is a fixed cascade; each stage caches its output and the fingerprint of the params that feed it:

```
StageCache {
    base:      Option<(BaseFingerprint, Heightmap)>,
    tectonics: Option<(TectonicsFingerprint, Heightmap)>,
    hydraulic: Option<(HydraulicFingerprint, Heightmap)>,
    // thermal output is the live heightmap; no need to cache beyond it
}
```

Fingerprints are `#[derive(PartialEq)]` structs holding exactly the param subset for that stage **plus the upstream fingerprint** (so dirtiness cascades structurally, not via flags):

| Fingerprint | Fields |
|---|---|
| `BaseFingerprint` | seed, grid, world_size, amplitude, octaves, frequency, persistence |
| `TectonicsFingerprint` | base: BaseFingerprint, uplift_strength, ridge_strength, uplift_frequency, ridge_frequency |
| `HydraulicFingerprint` | tectonics: TectonicsFingerprint, rain_rate, capacity_k, total_dig_budget |
| `ThermalFingerprint` | hydraulic: HydraulicFingerprint, talus_angle_deg |

`run_pipeline(gpu, params, cache) -> RegenOutput` computes the four fingerprints, finds the first stage whose cached fingerprint differs (or is absent), and re-runs from there, reusing upstream cached heightmaps. The cache lives CPU-side (4 × 2048² × 4 B ≈ 64 MB worst case — acceptable; cache entries for a different grid size are simply invalidated by the fingerprint mismatch).

Panel feedback after each run: which stages were reused vs run, plus per-stage timings (e.g. "reused fBm, tectonics · ran hydraulic 2.4 s, thermal 0.02 s").

Typical costs after the change: talus tweak ≈ 20 ms; erosion tuning ≈ 2.5 s; tectonic/fBm/seed/grid/size changes = full run (unchanged).

## 5. Export

The post-thermal `Heightmap` is retained in a `CurrentHeightmap` resource (it is already produced by every regen; today it is dropped after meshing).

Panel "Export" section → native folder picker (`rfd` crate, `FileDialog::pick_folder`) → writes:

| File | Format | Purpose |
|---|---|---|
| `height_16.png` | 16-bit grayscale PNG, min–max normalized | Unity / Unreal import |
| `height.r16` | raw little-endian u16, same normalization | Unreal RAW import |
| `height.exr` | 32-bit float, true world-unit heights | Blender / Gaea / Houdini |
| `biome_mask.png` | RGB, predicted-biome reference colours (reuses `analysis::predict_biome`) | splat-map source / inspection |
| `water_depth_16.png` | 16-bit grayscale, `max(0, sea_level − h)` normalized over its own range | shoreline / water masks |
| `metadata.json` | min/max heights, world size (units + metres), cell size, seed, full `GenParams` + `VisualParams` | scale reconstruction + reproducibility |

- EXR via the `exr` crate (single `height` float channel); PNG via the existing `image` crate; `.r16` is a plain byte write. All writers are pure functions `(heightmap, params, path) -> Result<(), String>` in a new `src/export.rs`.
- Export runs on the async task pool too (a 2048² EXR write is ~100 ms but the dialog + IO shouldn't touch the frame loop). Panel shows "exported to <path>" or the error string — no silent failure.

## 6. Verification protocol

**Objective — caching correctness (the load-bearing test):** `incremental_equals_full` in `src/erosion`/pipeline tests: for each stage boundary, run full pipeline at params P1, mutate one stage-k param to make P2, run incrementally (with cache) and from scratch (no cache); outputs must be **bit-identical**. Stages are individually deterministic, so any divergence indicts the fingerprint/dirty logic. Runs at 64² so it's fast enough for every test run.

**Objective — post-regen audit (in-app):** after every regen, run a cheap invariant audit on the new heightmap (all values finite; spike density < 0.5 % via `analysis::spike_mask`; relief within [50 %, 120 %] of the tectonic stage's relief — reusing the cached tectonics output). Panel shows ✓ or the violated invariant in red. Catches bad parameter combinations the moment they happen.

**Subjective — capture protocol:** `tools/verify.ps1`: runs `cargo nextest run`, then builds release and captures both reference viewpoints (close + wide) via `TERRAFORGE_CAPTURE`. Working rule (learned from the streak saga): after any visual-affecting change, run the script and compare both PNGs against the previous pair from the **same viewpoints** before claiming success. The export feature gives this protocol a third leg: exported `height_16.png` can be diffed numerically between runs.

## 7. Component boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `src/pipeline.rs` (new) | `GenParams` split into per-stage param structs, fingerprints, `StageCache`, cache-aware `run_pipeline` | `erosion`, `tectonics`, `terrain_noise`, `heightmap` |
| `src/export.rs` (new) | pure heightmap→file writers + metadata | `heightmap`, `analysis` (biome prediction), `image`, `exr` |
| `src/main.rs` | resources, UI panel, async task spawn/poll, visual-apply systems | `pipeline`, `export`, `biome` |
| `src/analysis.rs` | unchanged + post-regen audit helper | — |

`main.rs` is already ~550 lines; the pipeline and export extraction keeps it from becoming the god file.

## Error handling

- GPU errors during async regen surface as `Err(String)` from the task; the panel shows the message in red and the previous terrain stays untouched (no partial swaps).
- Export errors (permissions, disk) surface in the panel the same way.
- A regen requested while the GPU context is poisoned (device lost) re-creates the context once before failing loudly.

## Testing

- `incremental_equals_full` (per stage boundary, 64²) — the caching guard.
- `export` unit tests: PNG16 round-trip (encode→decode→compare within quantization), EXR round-trip (exact), `.r16` byte-length + endianness, metadata JSON parses and contains min/max/seed.
- Existing 33 tests must stay green; `pipeline_smoke` unchanged.
- Capture verification at defaults from both viewpoints must match the Phase 8 reference pair (the refactor must be pixel-neutral at defaults).

## Dependencies added

`rfd` (native dialogs), `exr` (EXR write), `serde`/`serde_json` (metadata). All mainstream, pure-Rust or windows-native.
