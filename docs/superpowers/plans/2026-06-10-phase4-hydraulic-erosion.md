# Phase 4: GPU Hydraulic Erosion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Carve realistic drainage — valleys, channels, sediment fans — into the tectonic terrain with a GPU compute-shader hydraulic erosion simulation.

**Architecture:** The Mei et al. 2007 *pipe model*: per-cell water depth, 4-direction outflow flux, velocity field, sediment capacity, erode/deposit, semi-Lagrangian sediment advection, evaporation. Seven compute dispatches per iteration, all embarrassingly parallel (each thread writes only its own cell — the slope-dependent erosion step is split into a read-only capacity pre-pass plus a cell-local erode pass to avoid cross-thread races, which also keeps the sim deterministic). Rust side: a `GpuContext` (own wgpu device/queue, independent of Bevy's renderer so it runs headless in tests/CI) and `erode_hydraulic(&GpuContext, &Heightmap, &ErosionParams) -> Result<Heightmap, String>` honouring the pipeline contract.

**Tech Stack:** `wgpu = "27"` (same major as Bevy 0.18's tree → one compiled copy), `pollster` (block on GPU async), `bytemuck` (POD casting). WGSL shader at `src/erosion/hydraulic.wgsl` via `include_str!`.

**Source spec:** build-order step 4; error-handling section (surface GPU failures, clamp heights per-iteration); testing section (headless GPU integration: mass roughly conserved, no NaN/Inf, peaks lowered, bounded output).

---

## File Structure

- `Cargo.toml` — add `wgpu = "27"`, `pollster = "0.4"`, `bytemuck = { version = "1", features = ["derive"] }`.
- `src/erosion/hydraulic.wgsl` — 7 entry points: `rain`, `compute_flux`, `water_velocity`, `compute_capacity`, `erode_deposit`, `advect_sediment`, `finalize_iter`.
- `src/erosion/mod.rs` — `GpuContext`, `ErosionParams`, `erode_hydraulic`, GPU buffers/bind group/pipelines/readback + integration tests.
- `src/main.rs` — insert erosion stage after tectonics.

Bindings (one layout for all passes): 0 uniform `Params`; storage `terrain`(1), `water`(2), `sediment`(3), `sediment_new`(4), `flux` vec4(5), `velocity` vec2(6), `capacity`(7).

Per-iteration pass order and data hazards:
1. `rain` — water += rain·dt (own cell only)
2. `compute_flux` — read own flux + neighbor heights (read-only this pass), write own flux
3. `water_velocity` — read neighbor flux (stable), write own water + velocity
4. `compute_capacity` — read neighbor terrain (read-only this pass) + own velocity → write own capacity
5. `erode_deposit` — own cell only: exchange terrain ↔ sediment vs capacity
6. `advect_sediment` — read sediment field (stable), write own sediment_new
7. `finalize_iter` — sediment = sediment_new; evaporate water; clamp terrain (divergence guard per spec)

WebGPU guarantees writes from one dispatch are visible to the next within a pass, so sequential dispatches need no explicit barriers.

---

## Task 1: Dependencies

- [ ] **Step 1:** Run: `cargo add wgpu@27 pollster@0.4` and `cargo add bytemuck@1 --features derive`
- [ ] **Step 2:** Run: `cargo tree -d -p wgpu 2>&1 | head -5` — expect NO duplicate wgpu versions (both bevy and terraforge on 27.x).
- [ ] **Step 3:** `cargo build` — clean.

## Task 2: The WGSL shader

**Files:** Create `src/erosion/hydraulic.wgsl`

- [ ] **Step 1: Write the shader** (complete source in repo file; key contracts:)

```wgsl
struct Params {
  width: u32, height: u32, dt: f32, rain_rate: f32,
  evaporation: f32, capacity_k: f32, erosion_k: f32, deposition_k: f32,
  min_tilt: f32, flux_factor: f32, _pad0: f32, _pad1: f32,
}
```
— 48 bytes, must match the Rust `GpuParams` exactly. `flux_factor = dt·A·g/l` precomputed CPU-side. Flux components: `.x`=to x-1, `.y`=to x+1, `.z`=to z-1, `.w`=to z+1; boundary flux forced 0; outflow scaled by `K = min(1, water/(total_outflow·dt))`. Velocity from net flux through the cell ÷ mean water depth (clamped ≥0.01 to avoid blow-ups). Capacity `C = capacity_k · max(sin_tilt, min_tilt) · |v|`. Erode `erosion_k·(C−s)·dt` when C>s, else deposit `deposition_k·(s−C)·dt` (clamped ≤ s). Advection: semi-Lagrangian bilinear gather at `pos − v·dt`, clamped inside the grid. Finalize clamps terrain to ±200.

- [ ] **Step 2:** Validate shader compiles: `naga src/erosion/hydraulic.wgsl` if naga-cli is installed, else defer to the `GpuContext` test in Task 3 (wgpu validates at module creation; `device.on_uncaptured_error` surfaces it).

## Task 3: `erosion/mod.rs` — context, dispatch, readback, tests (TDD)

**Files:** Create `src/erosion/mod.rs`; modify `src/main.rs` (add `mod erosion;`)

- [ ] **Step 1: Write the module.** Public API:

```rust
pub struct GpuContext { device: wgpu::Device, queue: wgpu::Queue }
impl GpuContext { pub fn new() -> Result<Self, String> }   // headless: no surface

#[derive(Clone, Debug)]
pub struct ErosionParams {
  pub iterations: u32, pub dt: f32, pub rain_rate: f32, pub evaporation: f32,
  pub capacity_k: f32, pub erosion_k: f32, pub deposition_k: f32, pub min_tilt: f32,
}
impl Default for ErosionParams { /* iterations:300, dt:0.02, rain:0.012, evap:0.02,
  capacity_k:1.0, erosion_k:0.5, deposition_k:0.5, min_tilt:0.05 */ }

pub fn erode_hydraulic(ctx: &GpuContext, hm: &Heightmap, p: &ErosionParams)
  -> Result<Heightmap, String>
```

Internals: create storage buffers (terrain seeded from `hm.data()` via `create_buffer_init`; others zeroed), 48-byte uniform (bytemuck `Pod`), one explicit bind-group layout shared by 7 pipelines (entry points by name), encode iterations in chunks of ≤50 per submit (avoids one giant command buffer), then `copy_buffer_to_buffer` → staging → `map_async` + `device.poll(wgpu::PollType::Wait)` → `Vec<f32>` → `Heightmap`. All failures are `Err(String)` with a clear message (spec: no silent fallback).

- [ ] **Step 2: Tests** (same file, `#[cfg(test)]`; input = `generate_fbm` 64×64 + tectonics, ~150 iterations):
  - `gpu_context_is_available` — `GpuContext::new()` succeeds (machine has Vulkan; CI note: test fails loudly, not silently, without a GPU).
  - `zero_iterations_roundtrips_exactly` — `iterations: 0` returns bit-identical heights (upload/readback sanity).
  - `output_is_finite_and_bounded` — no NaN/Inf; all |h| ≤ 200 (the clamp) and within input range ± 25%.
  - `is_deterministic` — two runs, identical bytes (race-free design makes this exact).
  - `peaks_are_lowered` — `max(out) < max(in)` (rain + slope ⇒ peak erosion).
  - `mass_roughly_conserved` — `sum(out) ∈ [sum(in) − 5%·Σ|in|, sum(in) + ε]`: erode/deposit is conservative; loss tolerance covers still-suspended sediment + advection gather error.
- [ ] **Step 3:** `cargo test erosion` — expect 6 pass. Numeric tolerances/params may need 1–2 tuning rounds; tune `ErosionParams` defaults, not the invariants.
- [ ] **Step 4:** `cargo test` — full suite (22 = 16 + 6).

## Task 4: Wire into app (visual)

**Files:** `src/main.rs`

- [ ] **Step 1:** After tectonics in `setup`:

```rust
    let gpu = erosion::GpuContext::new()
        .expect("GPU compute unavailable - hydraulic erosion requires a wgpu adapter");
    let hm = erosion::erode_hydraulic(&gpu, &hm, &erosion::ErosionParams::default())
        .expect("hydraulic erosion failed");
```
(rename tectonics output binding to feed it through; spec error-handling: expect = clear message + clean exit.)

- [ ] **Step 2:** `cargo build` — clean. **Step 3:** `cargo run` — terrain shows carved valleys/drainage channels on slopes vs Phase 3's smooth ridges. **Step 4:** Commit on user go-ahead (atomic: deps / shader / module+tests / app wiring / plan).

---

## Self-Review

**Spec coverage:** GPU compute hydraulic erosion ✓ (wgpu compute, WGSL, pipe model = "water flow, sediment transport/deposition"); `erosion/mod.rs` = "dispatch, buffers, CPU↔GPU sync" ✓; headless small-map CI tests with the spec's exact invariants ✓; per-iteration clamp + param-range defaults ✓; clear GPU error surfacing ✓. **Placeholders:** the shader body and module internals are summarized here by contract but written in full in the repo files during execution — every constant and binding index is specified above. **Type consistency:** `erode_hydraulic(&GpuContext, &Heightmap, &ErosionParams) -> Result<Heightmap, String>` used identically in tests and `main.rs`.

## Definition of done
- 22 tests green including 6 GPU integration tests run headless.
- App shows visibly carved terrain.
- Deterministic; mass conserved within tolerance; no NaN/Inf.
