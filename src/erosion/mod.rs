//! GPU hydraulic erosion: dispatch, buffers, CPU<->GPU sync.
//! Owns a headless wgpu device (independent of Bevy's renderer) so the
//! simulation runs identically in the app and in CI tests.

use wgpu::util::DeviceExt;

use crate::heightmap::Heightmap;

/// Headless GPU handle for compute work.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    /// Acquire a high-performance adapter and device. Errors are surfaced,
    /// never swallowed (spec: no silent fallback to a broken state).
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("no suitable GPU adapter for erosion compute: {e}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("terraforge-erosion"),
            ..Default::default()
        }))
        .map_err(|e| format!("GPU device request failed: {e}"))?;

        Ok(Self { device, queue })
    }
}

/// Tunables for the hydraulic erosion stage.
#[derive(Clone, Debug)]
pub struct ErosionParams {
    pub iterations: u32,
    /// Simulation time step (pipe-model stable region is roughly <= 0.05).
    pub dt: f32,
    /// Water added per cell per unit time.
    pub rain_rate: f32,
    /// Fraction of water evaporating per unit time.
    pub evaporation: f32,
    /// Sediment capacity constant Kc.
    pub capacity_k: f32,
    /// Dissolve rate Ks.
    pub erosion_k: f32,
    /// Deposition rate Kd.
    pub deposition_k: f32,
    /// Lower bound on sin(tilt) so flat-ish cells still transport a little.
    pub min_tilt: f32,
    /// Total normalized relief any single cell may carve over the whole run.
    /// Distributed as a per-iteration cap = total_dig_budget / iterations, so
    /// raising `iterations` makes the sim run longer without letting any one
    /// channel runaway-erode into a slot canyon.
    pub total_dig_budget: f32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        // Calibrated for heights normalized to [0,1] (erode_hydraulic
        // normalizes on upload and denormalizes on readback), following the
        // parameter regime of Mei et al. 2007.
        Self {
            iterations: 500,
            dt: 0.02,
            rain_rate: 0.012,
            evaporation: 0.05,
            capacity_k: 0.8,
            erosion_k: 0.5,
            deposition_k: 0.5,
            min_tilt: 0.02,
            total_dig_budget: 0.2,
        }
    }
}

/// Uniform block; layout must match `Params` in hydraulic.wgsl (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    dt: f32,
    rain_rate: f32,
    evaporation: f32,
    capacity_k: f32,
    erosion_k: f32,
    deposition_k: f32,
    min_tilt: f32,
    flux_factor: f32,
    dig_cap: f32,
    _pad0: f32,
}

// Dispatch order per iteration. advect_sediment runs before water_velocity
// because the flux-form advection needs the same water depths the flux
// K-clamp was computed against (water_velocity overwrites them).
const ENTRY_POINTS: [&str; 7] = [
    "rain",
    "compute_flux",
    "advect_sediment",
    "water_velocity",
    "compute_capacity",
    "erode_deposit",
    "finalize_iter",
];

/// Max erosion iterations encoded per queue submission, so one submit never
/// becomes a watchdog-sized command buffer.
const ITERATIONS_PER_SUBMIT: u32 = 50;

/// Run the pipe-model hydraulic erosion on the GPU.
/// Heightmap in -> heightmap out; deterministic for identical inputs.
pub fn erode_hydraulic(
    ctx: &GpuContext,
    hm: &Heightmap,
    params: &ErosionParams,
) -> Result<Heightmap, String> {
    let (w, h) = (hm.width, hm.height);
    let n = w * h;
    let device = &ctx.device;

    // The pipe model's parameter calibration assumes heights in ~[0,1]
    // (rain depth, flux and capacity are all coupled to that scale). Simulate
    // normalized, then map back to world units after readback.
    let in_min = hm.data().iter().cloned().fold(f32::INFINITY, f32::min);
    let in_max = hm.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (in_max - in_min).max(1e-6);
    let normalized: Vec<f32> = hm.data().iter().map(|v| (v - in_min) / range).collect();

    let gpu_params = GpuParams {
        width: w as u32,
        height: h as u32,
        dt: params.dt,
        rain_rate: params.rain_rate,
        evaporation: params.evaporation,
        capacity_k: params.capacity_k,
        erosion_k: params.erosion_k,
        deposition_k: params.deposition_k,
        min_tilt: params.min_tilt,
        // dt * A * g / l with A = 1, g = 9.81, l = 1.
        flux_factor: params.dt * 9.81,
        // Spread the dig budget across iterations so the carving contract is
        // honoured regardless of iteration count. Guard against /0.
        dig_cap: params.total_dig_budget / params.iterations.max(1) as f32,
        _pad0: 0.0,
    };

    // --- Buffers ---------------------------------------------------------
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("erosion-params"),
        contents: bytemuck::bytes_of(&gpu_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let terrain = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain"),
        contents: bytemuck::cast_slice(&normalized),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let make_zeroed = |label: &str, bytes_per_cell: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: n as u64 * bytes_per_cell,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false, // wgpu zero-initializes
        })
    };
    let water = make_zeroed("water", 4);
    let sediment = make_zeroed("sediment", 4);
    let sediment_new = make_zeroed("sediment-new", 4);
    let flux = make_zeroed("flux", 16);
    let velocity = make_zeroed("velocity", 8);
    let capacity = make_zeroed("capacity", 4);

    // --- Layout, pipelines, bind group ------------------------------------
    let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("erosion-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(1),
            storage_entry(2),
            storage_entry(3),
            storage_entry(4),
            storage_entry(5),
            storage_entry(6),
            storage_entry(7),
        ],
    });

    // Capture shader-compile/pipeline validation errors as a clear Err
    // instead of an uncaptured-error panic.
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hydraulic.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("hydraulic.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("erosion-pipeline-layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let make_pipeline = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipelines: Vec<wgpu::ComputePipeline> =
        ENTRY_POINTS.iter().map(|entry| make_pipeline(entry)).collect();
    // Runs once after the last iteration to ground suspended sediment.
    let settle_pipeline = make_pipeline("settle");

    if let Some(e) = pollster::block_on(device.pop_error_scope()) {
        return Err(format!("hydraulic erosion shader failed to build: {e}"));
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("erosion-bind-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: terrain.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: water.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sediment.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: sediment_new.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: flux.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: velocity.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: capacity.as_entire_binding() },
        ],
    });

    // --- Simulate ----------------------------------------------------------
    let wg_x = (w as u32).div_ceil(8);
    let wg_z = (h as u32).div_ceil(8);
    let mut remaining = params.iterations;
    while remaining > 0 {
        let batch = remaining.min(ITERATIONS_PER_SUBMIT);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("erosion-batch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("erosion"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            for _ in 0..batch {
                for pipeline in &pipelines {
                    pass.set_pipeline(pipeline);
                    pass.dispatch_workgroups(wg_x, wg_z, 1);
                }
            }
        }
        ctx.queue.submit(Some(encoder.finish()));
        remaining -= batch;
    }

    // Ground remaining suspended sediment so the terrain buffer is the full
    // mass ledger before readback.
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("erosion-settle"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("settle"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(&settle_pipeline);
            pass.dispatch_workgroups(wg_x, wg_z, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    // --- Read back ---------------------------------------------------------
    let byte_len = (n * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain-staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("erosion-readback"),
    });
    encoder.copy_buffer_to_buffer(&terrain, 0, &staging, 0, byte_len);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("GPU poll failed during readback: {e}"))?;
    rx.recv()
        .map_err(|_| "GPU readback callback dropped".to_string())?
        .map_err(|e| format!("mapping terrain staging buffer failed: {e:?}"))?;

    let heights: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    staging.unmap();

    Ok(Heightmap::from_fn(w, h, |x, z| in_min + heights[z * w + x] * range))
}

// ---------------------------------------------------------------------------
// Thermal erosion (angle-of-repose / talus smoothing)
// ---------------------------------------------------------------------------

/// Tunables for the GPU thermal-erosion stage.
#[derive(Clone, Debug)]
pub struct ThermalParams {
    pub iterations: u32,
    /// Max permitted height drop between 4-neighbours (in the heightmap's units).
    pub max_drop: f32,
    /// Fraction of the excess moved per iteration (0..1). Damping = 1.0 moves
    /// the full excess; smaller values are gentler.
    pub damping: f32,
}

impl Default for ThermalParams {
    fn default() -> Self {
        Self { iterations: 12, max_drop: 0.05, damping: 0.5 }
    }
}

/// Uniform block; must match `Params` in thermal.wgsl (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuThermalParams {
    width: u32,
    height: u32,
    max_drop: f32,
    damping: f32,
}

/// Max thermal iterations encoded per queue submission. Same rationale as the
/// hydraulic stage — one submit should never become a watchdog-sized batch.
const THERMAL_ITERATIONS_PER_SUBMIT: u32 = 50;

/// Run GPU talus / angle-of-repose smoothing on the heightmap. Heightmap in ->
/// heightmap out; deterministic for identical inputs.
pub fn erode_thermal(
    ctx: &GpuContext,
    hm: &Heightmap,
    params: &ThermalParams,
) -> Result<Heightmap, String> {
    let (w, h) = (hm.width, hm.height);
    let n = w * h;
    let device = &ctx.device;
    let byte_len = (n * 4) as u64;

    let gpu_params = GpuThermalParams {
        width: w as u32,
        height: h as u32,
        max_drop: params.max_drop,
        damping: params.damping,
    };

    // --- Buffers ---------------------------------------------------------
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("thermal-params"),
        contents: bytemuck::bytes_of(&gpu_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("thermal-terrain-a"),
        contents: bytemuck::cast_slice(hm.data()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("thermal-terrain-b"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- Layout, pipeline, bind groups -----------------------------------
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("thermal-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("thermal.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("thermal.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("thermal-pipeline-layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("talus_step"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("talus_step"),
        compilation_options: Default::default(),
        cache: None,
    });

    if let Some(e) = pollster::block_on(device.pop_error_scope()) {
        return Err(format!("thermal erosion shader failed to build: {e}"));
    }

    // Group A: read A, write B (used on even iterations 0, 2, 4, ...).
    // Group B: read B, write A (used on odd iterations 1, 3, 5, ...).
    let group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("thermal-bg-a"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_b.as_entire_binding() },
        ],
    });
    let group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("thermal-bg-b"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_a.as_entire_binding() },
        ],
    });

    // --- Simulate ----------------------------------------------------------
    let wg_x = (w as u32).div_ceil(8);
    let wg_z = (h as u32).div_ceil(8);
    let mut step: u32 = 0;
    let mut remaining = params.iterations;
    while remaining > 0 {
        let batch = remaining.min(THERMAL_ITERATIONS_PER_SUBMIT);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("thermal-batch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("thermal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            for _ in 0..batch {
                let bg = if step % 2 == 0 { &group_a } else { &group_b };
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_z, 1);
                step += 1;
            }
        }
        ctx.queue.submit(Some(encoder.finish()));
        remaining -= batch;
    }

    // After N iterations, the up-to-date buffer is A if N is even, else B.
    let output_buf = if params.iterations % 2 == 0 { &buf_a } else { &buf_b };

    // --- Read back ---------------------------------------------------------
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("thermal-staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("thermal-readback"),
    });
    encoder.copy_buffer_to_buffer(output_buf, 0, &staging, 0, byte_len);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("GPU poll failed during thermal readback: {e}"))?;
    rx.recv()
        .map_err(|_| "GPU thermal readback callback dropped".to_string())?
        .map_err(|e| format!("mapping thermal staging buffer failed: {e:?}"))?;

    let heights: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    staging.unmap();

    Ok(Heightmap::from_fn(w, h, |x, z| heights[z * w + x]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonics::{apply_tectonics, TectonicParams};
    use crate::terrain_noise::{generate_fbm, FbmParams};

    fn ctx() -> GpuContext {
        GpuContext::new().expect("erosion tests require a GPU adapter")
    }

    fn test_input() -> Heightmap {
        let base = generate_fbm(64, 64, &FbmParams::default());
        apply_tectonics(&base, &TectonicParams::default())
    }

    fn test_params() -> ErosionParams {
        ErosionParams::default()
    }

    #[test]
    fn gpu_context_is_available() {
        let _ = ctx();
    }

    #[test]
    fn zero_iterations_roundtrips_within_float_eps() {
        let hm = test_input();
        let out = erode_hydraulic(&ctx(), &hm, &ErosionParams { iterations: 0, ..Default::default() })
            .unwrap();
        // Heights are normalized to [0,1] for the sim and mapped back, so the
        // round-trip is float-exact only up to that affine transform.
        for (a, b) in out.data().iter().zip(hm.data()) {
            assert!((a - b).abs() < 1e-4, "roundtrip drifted: {b} -> {a}");
        }
    }

    #[test]
    fn output_is_finite_and_bounded() {
        let hm = test_input();
        let out = erode_hydraulic(&ctx(), &hm, &test_params()).unwrap();
        assert_eq!((out.width, out.height), (hm.width, hm.height));
        let in_min = hm.data().iter().cloned().fold(f32::INFINITY, f32::min);
        let in_max = hm.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let margin = 0.25 * (in_max - in_min);
        for &v in out.data() {
            assert!(v.is_finite(), "NaN/Inf in eroded terrain");
            assert!(v >= in_min - margin && v <= in_max + margin, "height {v} far outside input range");
        }
    }

    #[test]
    fn is_deterministic() {
        let hm = test_input();
        let c = ctx();
        let a = erode_hydraulic(&c, &hm, &test_params()).unwrap();
        let b = erode_hydraulic(&c, &hm, &test_params()).unwrap();
        assert_eq!(a.data(), b.data(), "race-free pipeline must be bit-deterministic");
    }

    #[test]
    fn peaks_are_lowered() {
        let hm = test_input();
        let out = erode_hydraulic(&ctx(), &hm, &test_params()).unwrap();
        // The single argmax cell is a flow stagnation point (symmetric
        // drainage => zero net velocity => zero capacity), so hydraulic
        // erosion can leave it bit-identical; flattening summits is thermal
        // erosion's job (Phase 5). The hydraulic invariants are:
        // (a) nothing may RAISE the summit, and (b) the high terrain as a
        // whole (top 1% of cells) must lose material.
        let max_in = hm.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_out = out.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_out <= max_in, "deposition must not raise the summit ({max_in} -> {max_out})");

        let top_percentile_mean = |hm: &Heightmap| {
            let mut v: Vec<f32> = hm.data().to_vec();
            v.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let k = (v.len() / 100).max(1);
            v[..k].iter().sum::<f32>() / k as f32
        };
        let top_in = top_percentile_mean(&hm);
        let top_out = top_percentile_mean(&out);
        assert!(
            top_out < top_in,
            "top-1% of terrain should be lowered by erosion ({top_in} -> {top_out})"
        );
    }

    #[test]
    fn mass_roughly_conserved() {
        let hm = test_input();
        let out = erode_hydraulic(&ctx(), &hm, &test_params()).unwrap();
        let sum_in: f32 = hm.data().iter().sum();
        let sum_out: f32 = out.data().iter().sum();
        let scale: f32 = hm.data().iter().map(|v| v.abs()).sum();
        let drift = (sum_out - sum_in).abs() / scale.max(1.0);
        assert!(drift < 0.05, "terrain mass drifted {:.2}% (suspended sediment tolerance is 5%)", drift * 100.0);
    }

    // ----- Thermal erosion -------------------------------------------------

    fn thermal_params() -> ThermalParams {
        ThermalParams::default()
    }

    /// 16x16 heightmap with a single spike at the centre.
    fn spike_input() -> Heightmap {
        let mut hm = Heightmap::new(16, 16);
        hm.set(8, 8, 10.0);
        hm
    }

    #[test]
    fn thermal_zero_iterations_roundtrips() {
        let hm = test_input();
        let out = erode_thermal(&ctx(), &hm, &ThermalParams { iterations: 0, ..Default::default() })
            .unwrap();
        // Even-iter output reads from buf_a (the upload buffer), so a 0-iter
        // run is exact byte-for-byte — no normalization round-trip like the
        // hydraulic stage.
        assert_eq!(out.data(), hm.data(), "0-iter thermal must roundtrip exactly");
    }

    #[test]
    fn thermal_is_deterministic() {
        let hm = test_input();
        let c = ctx();
        let a = erode_thermal(&c, &hm, &thermal_params()).unwrap();
        let b = erode_thermal(&c, &hm, &thermal_params()).unwrap();
        assert_eq!(a.data(), b.data(), "race-free pipeline must be bit-deterministic");
    }

    #[test]
    fn thermal_conserves_mass() {
        let hm = test_input();
        let out = erode_thermal(&ctx(), &hm, &thermal_params()).unwrap();
        let sum_in: f32 = hm.data().iter().sum();
        let sum_out: f32 = out.data().iter().sum();
        let scale: f32 = hm.data().iter().map(|v| v.abs()).sum();
        let drift = (sum_out - sum_in).abs() / scale.max(1.0);
        // Pair-wise exchange is exact in algebra; drift is pure f32 rounding.
        assert!(drift < 1e-3, "thermal mass drifted {:.4}% (target < 0.1%)", drift * 100.0);
    }

    #[test]
    fn thermal_collapses_single_spike() {
        let hm = spike_input();
        let p = ThermalParams { iterations: 4, max_drop: 0.5, damping: 0.5 };
        let out = erode_thermal(&ctx(), &hm, &p).unwrap();
        let spike_h = out.get(8, 8);
        // Talus invariant: after relaxation the spike is no more than `max_drop`
        // above its 4-neighbours. (At damping=0.5 the operator is exactly at
        // the stability boundary, so the local steady-state is a small cone
        // rather than full diffusion — but the angle constraint must hold.)
        let mut max_neighbor = f32::NEG_INFINITY;
        for (nx, nz) in [(7, 8), (9, 8), (8, 7), (8, 9)] {
            let h = out.get(nx, nz);
            assert!(h > 0.0, "neighbour ({nx},{nz}) should have gained mass (got {h})");
            if h > max_neighbor { max_neighbor = h; }
        }
        // 1e-4 slack for f32 rounding in the gradient comparison.
        assert!(
            spike_h - max_neighbor <= p.max_drop + 1e-4,
            "talus angle violated: spike {spike_h} is {} above max-neighbour {max_neighbor} (max_drop={})",
            spike_h - max_neighbor, p.max_drop
        );
        // Spike must also have lost substantial height — sanity check that the
        // pass actually moved material rather than being a no-op.
        assert!(spike_h < 5.0, "spike (10.0) must lose >50% mass to neighbours (got {spike_h})");
    }

    #[test]
    fn thermal_preserves_gentle_ramp() {
        // Slope = 0.5 * max_drop per cell, well below the talus angle, so the
        // pass must leave the field bit-identical (no diffusion creep).
        let max_drop = 0.5_f32;
        let hm = Heightmap::from_fn(16, 16, |x, _z| x as f32 * (0.5 * max_drop));
        let p = ThermalParams { iterations: 8, max_drop, damping: 0.5 };
        let out = erode_thermal(&ctx(), &hm, &p).unwrap();
        for (a, b) in out.data().iter().zip(hm.data()) {
            assert!((a - b).abs() < 1e-5, "ramp below talus must stay put ({b} -> {a})");
        }
    }
}
