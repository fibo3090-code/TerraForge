//! Phase 6 biome material: `ExtendedMaterial<StandardMaterial, BiomeExtension>`.
//! Plumbing only — the shading rules live in `assets/shaders/biome.wgsl`.

use bevy::{
    asset::{Asset, Handle, RenderAssetUsages},
    image::{Image, ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{
        AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
    },
    shader::ShaderRef,
};
use image::imageops::FilterType;
use std::path::Path;

/// The fully-typed material we spawn on the terrain mesh.
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, BiomeExtension>;

/// Uniform block — must match `BiomeParams` in `assets/shaders/biome.wgsl`.
#[derive(Copy, Clone, Debug, ShaderType)]
#[repr(C)]
pub struct BiomeParams {
    /// World-space Y at which `h_norm` is 0 (typically `heightmap.min`).
    pub world_y_min: f32,
    /// World-space Y range over which `h_norm` sweeps 0..1.
    pub world_y_range: f32,
    /// Normalized elevation (0..1) at the centre of the snow band.
    pub snow_line: f32,
    /// Smoothstep half-width on the snow elevation axis.
    pub snow_blend: f32,
    /// cos(rock-slope-angle): a surface is "steep" iff `world_normal.y < this`.
    pub rock_slope_cos: f32,
    /// Smoothstep half-width on the slope axis.
    pub rock_blend: f32,
    /// Normalized elevation at the grass<->dirt transition.
    pub grass_dirt_line: f32,
    pub grass_dirt_blend: f32,
    /// Planar UV scale (UV repeats per world unit) for grass/dirt.
    pub uv_scale_planar: f32,
    /// Triplanar UV scale for rock/snow.
    pub uv_scale_triplanar: f32,
    /// How strongly the detail normal is blended into the surface normal (0..1).
    pub normal_strength: f32,
    pub _pad0: f32,
}

impl Default for BiomeParams {
    fn default() -> Self {
        // Defaults are normalized — the loader overrides world_y_min/range
        // with the actual heightmap extent.
        Self {
            world_y_min: 0.0,
            world_y_range: 1.0,
            snow_line: 0.62,
            snow_blend: 0.10,
            rock_slope_cos: 32.0_f32.to_radians().cos(),
            rock_blend: 0.15,
            // Higher line + wider blend so grass dominates the low/mid hills
            // and dirt only takes over closer to the rocky ridges. Combined
            // with the macro-noise modulation in the shader this reads as
            // patchy ground rather than clean elevation bands.
            grass_dirt_line: 0.45,
            grass_dirt_blend: 0.20,
            uv_scale_planar: 1.0 / 8.0,
            uv_scale_triplanar: 1.0 / 6.0,
            // Gentle. With proper mipmap chains on the normal maps the
            // glancing-angle UV-aliasing streaks (that an earlier session
            // misdiagnosed as a normal-map issue) are gone; 0.15 puts back
            // micro-detail without over-warping the normal.
            normal_strength: 0.15,
            _pad0: 0.0,
        }
    }
}

#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct BiomeExtension {
    #[uniform(100)]
    pub params: BiomeParams,
    // grass_albedo carries the shared sampler at slot 109 — Bevy's AsBindGroup
    // requires `#[sampler]` to ride along with a `#[texture]` on the same
    // Handle<Image>. The WGSL only references this binding via `biome_sampler`.
    #[texture(101)]
    #[sampler(109)]
    pub grass_albedo: Handle<Image>,
    #[texture(102)]
    pub grass_normal: Handle<Image>,
    #[texture(103)]
    pub dirt_albedo: Handle<Image>,
    #[texture(104)]
    pub dirt_normal: Handle<Image>,
    #[texture(105)]
    pub rock_albedo: Handle<Image>,
    #[texture(106)]
    pub rock_normal: Handle<Image>,
    #[texture(107)]
    pub snow_albedo: Handle<Image>,
    #[texture(108)]
    pub snow_normal: Handle<Image>,
}

impl MaterialExtension for BiomeExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/biome.wgsl".into()
    }
}

/// Load the 8 biome textures and return a ready-to-use `TerrainMaterial`.
/// Albedo loads default to sRGB (correct); normals override the loader
/// to treat the bytes as linear (`Rgba8Unorm` after decode).
/// Load a JPG from disk and pack it into a Bevy `Image` with a full mipmap
/// chain. The base JPG ships with only the top level — without mips, the
/// GPU samples one texel per fragment at glancing angles and the result is
/// severe directional UV-aliasing streaks across slopes (see
/// `test_output/16_yplane_only.png` for what an un-mipped 1024² texture
/// looks like on a close-up slope). Anisotropic filtering is meaningless
/// without mip levels to interpolate between.
fn load_image_with_mips(absolute_path: &Path, is_srgb: bool) -> Result<Image, String> {
    let dyn_img = image::open(absolute_path)
        .map_err(|e| format!("open {}: {e}", absolute_path.display()))?
        .to_rgba8();
    let (w, h) = (dyn_img.width(), dyn_img.height());

    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(dyn_img.as_raw());

    // Generate down-sampled mip levels (Lanczos3) until 1×1.
    let mut current = dyn_img;
    let mut mip_count: u32 = 1;
    while current.width() > 1 && current.height() > 1 {
        let next_w = (current.width() / 2).max(1);
        let next_h = (current.height() / 2).max(1);
        let next = image::imageops::resize(&current, next_w, next_h, FilterType::Triangle);
        data.extend_from_slice(next.as_raw());
        mip_count += 1;
        current = next;
    }

    let format = if is_srgb { TextureFormat::Rgba8UnormSrgb } else { TextureFormat::Rgba8Unorm };
    let mut img = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        format,
        RenderAssetUsages::default(),
    );
    img.texture_descriptor.mip_level_count = mip_count;
    // Repeat addressing is the load-bearing setting here: Bevy samplers
    // default to ClampToEdge, and our world-space UVs span many repeats —
    // with clamping, the texture renders only inside the single 0..1 UV tile
    // and everywhere else its edge texels smear into long directional streaks
    // (the "lines" artifact). Anisotropic linear + mips for quality.
    let mut sampler = ImageSamplerDescriptor::linear();
    sampler.address_mode_u = ImageAddressMode::Repeat;
    sampler.address_mode_v = ImageAddressMode::Repeat;
    sampler.anisotropy_clamp = 16;
    img.sampler = ImageSampler::Descriptor(sampler);
    Ok(img)
}

pub fn build_terrain_material(
    _asset_server: &AssetServer,
    materials: &mut Assets<TerrainMaterial>,
    images: &mut Assets<Image>,
    params: BiomeParams,
) -> Handle<TerrainMaterial> {
    // Load each biome JPG ourselves so we can attach a full mipmap chain —
    // Bevy's asset loader doesn't generate mips and the missing chain causes
    // glancing-angle UV aliasing to render as severe directional streaks.
    let root = std::env::var("BEVY_ASSET_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default())
        .join("assets/biomes");
    let mut load = |name: &str, srgb: bool| -> Handle<Image> {
        let p = root.join(name);
        match load_image_with_mips(&p, srgb) {
            Ok(img) => images.add(img),
            Err(e) => panic!("biome texture load failed: {e}"),
        }
    };

    let grass_albedo = load("grass_albedo.jpg", true);
    let grass_normal = load("grass_normal.jpg", false);
    let dirt_albedo = load("dirt_albedo.jpg", true);
    let dirt_normal = load("dirt_normal.jpg", false);
    let rock_albedo = load("rock_albedo.jpg", true);
    let rock_normal = load("rock_normal.jpg", false);
    let snow_albedo = load("snow_albedo.jpg", true);
    let snow_normal = load("snow_normal.jpg", false);

    let ext = BiomeExtension {
        params,
        grass_albedo,
        grass_normal,
        dirt_albedo,
        dirt_normal,
        rock_albedo,
        rock_normal,
        snow_albedo,
        snow_normal,
    };
    materials.add(ExtendedMaterial {
        base: StandardMaterial {
            // Roughness/metallic are tuned for terrain; the shader still drives
            // the colour and perturbed normal.
            perceptual_roughness: 0.92,
            metallic: 0.0,
            reflectance: 0.05,
            ..default()
        },
        extension: ext,
    })
}
