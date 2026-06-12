# Milestone 2B: Hydrology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Rivers and lakes via static DEM analysis — priority-flood depression filling (lakes + fill levels) and D8 flow accumulation (river networks) — with river-bed carving as a 5th cached pipeline stage and a rendered water surface mesh.

**Architecture:** `src/hydrology.rs` is pure CPU (`Heightmap` in → carved `Heightmap` + `WaterField{surface, depth}` out), fully testable headless. `pipeline.rs` gains stage 5 (`HydrologySettings` in `PipelineParams`, thermal output joins the cache, fingerprint cascade extended). Rendering: a second water mesh built from the `WaterField` (lakes at their fill levels, rivers half-full in carved channels) sharing the ocean's water material. Export gains real water depth + water mask.

**Decisions (user):** static analysis (not animated sim); carve river beds (5th stage, in the cache/fingerprint system).

## Tasks

### Task 1: `src/hydrology.rs` (TDD)
- [ ] Priority-flood (Barnes, epsilon variant) seeded from borders; `fill >= terrain` invariant; bowl-with-notch fills to spill height.
- [ ] D8 flow accumulation over the filled surface (highest→lowest order, steepest-descent routing); ramp test: accumulation grows downslope, bottom row drains its column.
- [ ] `run_hydrology`: river cells where `acc >= river_threshold * n`; radial parabolic carve scaled by log-drainage (max `carve_depth`, half-width `river_width` cells); lakes = `filled > terrain`; rivers run half-full. Tests: carve bounded + localized + never raises; water surface above bed; bowl contains a lake.
- [ ] Commit: `feat: DEM hydrology — priority-flood lakes + D8 rivers + carving`

### Task 2: pipeline stage 5
- [ ] `PipelineParams.hydrology: HydrologySettings`; `StageCache.thermal` entry added (keyed by all upstream params); `run_pipeline` runs hydrology after thermal, returns carved heightmap + `water: WaterField` + river/lake counts; progress gains HYDROLOGY=4 (MESH=5, DONE=6).
- [ ] Tests: `incremental_equals_full` gains a hydrology mutation; `thermal_change_reuses_upstream` now expects `stages_run == [thermal, hydrology]`; new `hydrology_change_reuses_all_four`.
- [ ] Commit: `feat: hydrology as 5th cached pipeline stage`

### Task 3: rendering + UI + export + audit
- [ ] Water surface mesh: build a `Heightmap` (wet → surface elevation, dry → terrain − 0.3) downsampled to ≤1024², through `heightmap_to_mesh`, spawned with the shared water material; rebuilt in `poll_regen` via `meshes.insert`.
- [ ] Panel Generation section gains river density (log slider 0.0002–0.01), carve depth (0–1), river width (0–6); LastRunInfo shows river/lake cell counts.
- [ ] Audit spike threshold becomes `max(2*max_drop, carve_depth * 1.2)` (carved channels are intentional steps).
- [ ] `export_all` takes `Option<&WaterField>`: real `water_depth_16.png` + new `water_mask.png`; tests updated (None → 6 files, Some → 7).
- [ ] Commit: `feat: water surface mesh, hydrology controls, water exports`

### Task 4: verify
- [ ] `cargo nextest run` all green; `tools/verify.ps1`; compare both captures vs previous pair — wide view must now show rivers/lakes; close view should be near-identical except where channels carve.
- [ ] Commit any capture-driven tuning.

## Definition of done
- ~50 tests green incl. 5 hydrology unit tests + extended cache tests.
- Wide capture shows river ribbons feeding the lake/sea; no spike-audit false positives from carving.
- Talus/hydrology slider changes re-run in <1.5 s at 2048² (thermal+hydrology only).
