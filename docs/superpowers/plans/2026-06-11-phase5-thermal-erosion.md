# Phase 5: GPU Thermal Erosion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the CPU talus stopgap with a GPU thermal-erosion stage that holds slopes at the angle of repose, runs through the shared `GpuContext`, and is deterministic + headless-testable on the same terms as hydraulic erosion.

**Architecture:** Per-cell pipe-model analogue for dry granular flow. Each thread reads its own height and its 4 neighbors from `terrain_in`; for every neighbor lower than self by more than `max_drop`, it sheds `(excess − max_drop)/2` into a private accumulator and writes the result to `terrain_out`. Mirror logic on the receive side keeps the pair-wise exchange mass-conservative to f32 rounding. Multiple iterations ping-pong `terrain_in`/`terrain_out` each step so writes from iteration *k* are visible as reads in iteration *k+1* without explicit barriers. The CPU talus pass in `src/talus.rs` (a stopgap added during Phase 4 polish) is the reference implementation: same algorithm, fewer rounding optimisations, exhaustive tests.

**Tech Stack:** Existing `wgpu = "27"` + `pollster` + `bytemuck`. New shader at `src/erosion/thermal.wgsl` via `include_str!`. No new dependencies.

**Source spec:** build-order step 5 (`thermal erosion (GPU) ← talus slopes / angle of repose`); file structure lists `src/erosion/thermal.wgsl`; error handling (surface GPU failures, clamp output); testing (mass conservation, idempotent on a ramp, single spike collapses, deterministic).

---

## File Structure

- `src/erosion/thermal.wgsl` — single entry point `talus_step` (ping-pong, 8×8 workgroup, 4-neighbour).
- `src/erosion/mod.rs` — add `ThermalParams`, `erode_thermal(&GpuContext, &Heightmap, &ThermalParams) -> Result<Heightmap, String>`; share buffer/error-scope idioms with `erode_hydraulic`. Add 4 integration tests mirroring the Phase 4 invariants.
- `src/main.rs` — swap the CPU `apply_talus` call for `erosion::erode_thermal`; keep the same `max_drop` derivation from cell size + `TALUS_ANGLE_DEG`.
- `src/talus.rs` — retained for the unit tests (algorithmic reference); the production pipeline no longer calls it. Marked `#[allow(dead_code)]` until Phase 6 removes the import from `main.rs`.

Bindings (one layout, ping-pong inputs):
```
@binding(0) uniform Params { width, height, max_drop, damping, _pad }   // 16 B
@binding(1) storage<read>       terrain_in : array<f32>
@binding(2) storage<read_write> terrain_out: array<f32>
```

Two bind groups created once: group A reads buffer X writes buffer Y; group B reads Y writes X. Loop binds A, dispatch, binds B, dispatch — no buffer copies, no barriers needed (storage writes from a finished dispatch are visible to the next dispatch's reads within the same encoder, per the WebGPU spec).

---

## Task 1: The WGSL shader

**Files:** Create `src/erosion/thermal.wgsl`

- [ ] **Step 1:** Write the shader. Key contracts:

```wgsl
struct Params {
    width: u32,
    height: u32,
    max_drop: f32,
    damping: f32,   // 0..1 fraction of excess moved per iteration
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       terrain_in:  array<f32>;
@group(0) @binding(2) var<storage, read_write> terrain_out: array<f32>;

fn idx(x: u32, z: u32) -> u32 { return z * p.width + x; }

fn contribution(me: f32, neighbour: f32) -> f32 {
    let diff = neighbour - me;
    if (diff >  p.max_drop) { return (diff - p.max_drop) * p.damping * 0.5; }
    if (diff < -p.max_drop) { return (diff + p.max_drop) * p.damping * 0.5; }
    return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn talus_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) { return; }
    let x = gid.x; let z = gid.y; let i = idx(x, z);
    let me = terrain_in[i];
    var delta = 0.0;
    if (x > 0u)            { delta += contribution(me, terrain_in[idx(x - 1u, z)]); }
    if (x + 1u < p.width)  { delta += contribution(me, terrain_in[idx(x + 1u, z)]); }
    if (z > 0u)            { delta += contribution(me, terrain_in[idx(x, z - 1u)]); }
    if (z + 1u < p.height) { delta += contribution(me, terrain_in[idx(x, z + 1u)]); }
    terrain_out[i] = me + delta;
}
```

Pair-wise symmetry note: cell A's `contribution(A, B)` with `B − A > max_drop` evaluates to `(B − A − max_drop) · damping · 0.5` (A receives); the same iteration evaluates B's `contribution(B, A)` as `−(B − A − max_drop) · damping · 0.5` (B sheds). Equal magnitude, opposite sign → mass conserved exactly per pair, to f32 rounding.

- [ ] **Step 2:** Defer compile validation to the wgpu validation error scope opened by `erode_thermal` (same idiom as `erode_hydraulic`).

## Task 2: `erode_thermal` (TDD)

**Files:** Modify `src/erosion/mod.rs`

- [ ] **Step 1:** Add params:

```rust
#[derive(Clone, Debug)]
pub struct ThermalParams {
    pub iterations: u32,
    /// Max permitted height drop between 4-neighbours (world units of the input heightmap).
    pub max_drop: f32,
    /// Fraction of the excess moved per iteration (0..1).
    pub damping: f32,
}

impl Default for ThermalParams {
    fn default() -> Self {
        Self { iterations: 12, max_drop: 0.05, damping: 0.5 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuThermalParams {
    width: u32, height: u32, max_drop: f32, damping: f32,
}
```

- [ ] **Step 2:** Add the function:

```rust
pub fn erode_thermal(
    ctx: &GpuContext,
    hm: &Heightmap,
    params: &ThermalParams,
) -> Result<Heightmap, String>
```

Implementation pattern (mirror `erode_hydraulic`):
1. Two `STORAGE | COPY_SRC` f32 buffers (`buf_a` seeded from `hm.data()`, `buf_b` zero-initialised).
2. One uniform buffer from `GpuThermalParams`.
3. One bind-group layout with the 3 bindings above.
4. **Two** bind groups: A = read `buf_a` / write `buf_b`; B = read `buf_b` / write `buf_a`.
5. Push validation error scope, build shader + compute pipeline (entry `talus_step`), pop and surface compile errors as `Err(String)`.
6. Per submit, dispatch up to 50 iterations alternating bind groups (so 1 submit ≤ 50 ms of GPU work).
7. Final output buffer is `buf_a` if `iterations` is even, else `buf_b` — readback that one via staging + `map_async` + `device.poll(PollType::wait_indefinitely())`.

Workgroup count: `(width.div_ceil(8), height.div_ceil(8), 1)`. Cell `l = 1` in heightmap coordinates; the caller supplies `max_drop` already scaled to whatever units the heightmap uses.

- [ ] **Step 3:** Add 4 integration tests (mirror Phase 4 style):

  - `thermal_zero_iterations_roundtrips` — `iterations: 0` returns bit-identical heights (upload/readback sanity; even iterations → starts and ends in `buf_a`).
  - `thermal_is_deterministic` — two runs at identical params produce byte-identical output (race-free design must be exact).
  - `thermal_conserves_mass` — `sum(out)` within 1e-3 × `sum(|in|)` of `sum(in)`; pair-wise exchange is exact in algebra, drift is pure f32 rounding.
  - `thermal_collapses_single_spike` — a 16×16 zero field with one 10.0 cell in the centre: after 4 iterations the centre cell `< 1.0` AND each of its 4 neighbours `> 0.0` (spike was redistributed, not just clipped).
  - `thermal_preserves_gentle_ramp` — ramp at 0.5 × `max_drop` per cell: output equals input to 1e-5 (slope below talus must not move).

- [ ] **Step 4:** `cargo nextest run erosion::tests::thermal` — expect 5 pass. If `thermal_conserves_mass` drifts more than 1e-3 × |in|, audit the contribution function for sign / boundary asymmetry — that is the only way mass leaks in a symmetric algorithm.

- [ ] **Step 5:** `cargo nextest run` — full suite (current 26 + 5 thermal = 31).

## Task 3: Wire into `main.rs`

**Files:** Modify `src/main.rs`

- [ ] **Step 1:** Replace the CPU talus call:

```rust
let t = std::time::Instant::now();
let hm = erosion::erode_thermal(
    &gpu,
    &hm,
    &erosion::ThermalParams { iterations: TALUS_ITERATIONS, max_drop, damping: 0.5 },
)
.expect("thermal erosion failed");
info!(
    "thermal erosion (max_drop={max_drop:.3}, {TALUS_ITERATIONS} iter): {:?}",
    t.elapsed()
);
```

- [ ] **Step 2:** Drop the `use talus::{apply_talus, TalusParams}` import; leave `mod talus;` (its unit tests stay as the algorithmic reference) and silence the dead-code warnings with `#[allow(dead_code)]` on `apply_talus`/`TalusParams`. Phase 6 removes the module entirely if no one is reading it.

- [ ] **Step 3:** `cargo build --release` — clean.
- [ ] **Step 4:** `cargo run --release` — the heightmap stats line should still show `range ≈ 35–50` (i.e. thermal hasn't washed the relief away) and visually the terrain matches the previous CPU-talus output ± float noise. The timing line for the new stage should be sub-30 ms at 2048² with 12 iterations (≈ 50× faster than CPU).
- [ ] **Step 5:** Commit on user go-ahead (atomic: shader / module + tests / app wiring / plan).

---

## Self-Review

**Spec coverage:** GPU thermal erosion ✓ (compute pass, talus / angle of repose); `erosion/thermal.wgsl` ✓; headless tests with mass / ramp / spike invariants matching Phase 4's discipline ✓; clear `Result<_, String>` error surface (no silent fallback) ✓. **Placeholders:** the shader body above is final; the Rust function's wiring is described by contract and mirrors the well-debugged `erode_hydraulic` pattern in the same file. **Type consistency:** `erode_thermal(&GpuContext, &Heightmap, &ThermalParams) -> Result<Heightmap, String>` used identically in tests and `main.rs`. **CPU talus stopgap:** retained as `src/talus.rs` for its unit tests (independent algorithmic check) but no longer on the hot path.

## Definition of done
- 31 tests green including 5 new thermal integration tests run headless.
- App's pipeline runs hydraulic → GPU thermal (no CPU talus pass).
- Output heightmap shows the same overall character as the CPU-talus version (heights ± float noise; same min/max/mean to within rounding).
- New stage timing < 30 ms at 2048² with 12 iterations.
- Deterministic; mass conserved within tolerance; no NaN/Inf.
