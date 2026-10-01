# TerraForge

Real-time, fly-around 3D **realistic terrain generator** in Rust + Bevy 0.18. A
five-stage simulation pipeline (fBm noise → tectonics → GPU hydraulic erosion →
GPU thermal erosion → DEM hydrology) produces mountains carved by drainage
networks, lakes at their natural spill levels, biome texturing by slope and
altitude, and a physically-based sky — all editable live from an in-app panel
and exportable to game engines and DCC tools.

## Quick start

Requirements: Rust (stable), a Vulkan-capable GPU, `cargo-nextest`
(`cargo install cargo-nextest`).

```powershell
cargo run --release        # opens the workbench (~1 s to first terrain)
cargo nextest run          # full test suite (needs the GPU)
powershell tools/verify.ps1  # tests + dual-viewpoint render captures
```

## Controls

| Input | Action |
|---|---|
| Mouse drag / wheel | Orbit / zoom (panorbit camera) |
| `R` | Regenerate terrain |
| `F1` | Hide/show the panel |
| Panel → Generation | Seed, grid res, map size (1–10 km), noise/tectonic/erosion/river params — applied on **Regenerate** (async; the UI never blocks) |
| Panel → Environment | Sea level, snow line, rock slope, sun, water colour — applied **instantly** |
| Panel → Export | Writes `height_16.png` / `height.r16` (Unreal/Unity), `height.exr` (Blender/Gaea), biome + water masks, `metadata.json` (scales + all params for reproducibility) |

## Architecture (for new contributors)

```
src/
  terrain_noise.rs   fBm Perlin base            (CPU, pure)
  tectonics.rs       uplift + ridge guidance     (CPU, pure)
  erosion/           hydraulic + thermal erosion (GPU compute, WGSL, headless wgpu)
  hydrology.rs       priority-flood lakes + D8 rivers + carving (CPU, pure)
  pipeline.rs        the 5-stage cascade with per-stage caching — a param
                     change re-runs only the stages it affects, and
                     `incremental_equals_full` proves cached == cold, bitwise
  mesh_builder.rs    Heightmap -> Bevy mesh
  biome.rs +         slope/altitude biome texturing
  assets/shaders/    (triplanar, ExtendedMaterial on StandardMaterial)
  hydrology water    rendered as a surface mesh hugging lakes/rivers
  export.rs          pure heightmap -> file writers
  analysis.rs        verification harness: stats, spike masks, biome
                     prediction, streak detection, diagnostic PNG exports
  main.rs            Bevy app: resources, egui panel, async regen plumbing
```

Core contract: **every stage is heightmap-in → heightmap-out**, so stages
compose and any stage can be replaced (e.g. a future ML base layer).

## Verification culture

Quality is enforced by tooling, not vigilance — see [docs/QUALITY.md](docs/QUALITY.md):

- `cargo nextest run` — ~50 tests incl. GPU integration tests and the
  pipeline-cache bit-identity guarantee.
- Every regen runs an in-app invariant audit (finite heights, spike density,
  relief preservation) shown in the panel.
- `tools/verify.ps1` — one command: full suite + GPU captures from the two
  reference viewpoints. **Compare captures against the previous pair from the
  same viewpoints before claiming a visual change is safe** — artifacts hide
  at the viewpoint you didn't check.

## Licences

Code: MIT — see [LICENSE](LICENSE). Biome textures: CC0 from [ambientCG](https://ambientcg.com)
(see `assets/biomes/README.md`).
