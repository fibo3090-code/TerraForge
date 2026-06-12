# TerraForge Quality Protocol

The standing workflow for keeping the app fast, usable, and verifiable — for
users and for everyone working on the code. Run through this checklist before
calling any change done; it is the institutional version of lessons this
project already paid for once.

## Metrics we hold the line on

| Metric | Target | How to measure |
|---|---|---|
| Time to first terrain on launch | < 1.5 s | log line `pipeline: 512x512 …` + window appears |
| Full 2048² regen (cold cache) | < 8 s, UI responsive throughout | panel timing line after Regenerate |
| Talus/river param regen (warm cache) | < 1.5 s | panel shows `reused fBm, tectonics, hydraulic…` |
| Frame rate while orbiting (RTX-class GPU) | ≥ 60 fps | fps readout at the panel footer |
| Test suite | 100 % green, < 60 s wall | `cargo nextest run` |
| Clippy | 0 warnings | `cargo clippy` |
| Export correctness | metadata.json reconstructs real-world scale | open exported folder in target tool |

## The verification loop (run after every change)

1. **Objective:** `cargo nextest run`. The non-negotiables:
   - `incremental_equals_full` — pipeline cache produces bit-identical output.
   - erosion invariants (mass conservation, determinism, bounded output).
   - hydrology invariants (fill ≥ terrain, bounded carve, lakes at spill level).
   - export round-trips (PNG16 within quantization, EXR exact).
2. **Objective, in-app:** every regen runs the invariant audit (finite, spike
   density < 0.5 %, relief within [50 %, 120 %] of tectonic relief). Red text
   in the panel = a bad parameter regime; investigate before shipping defaults.
3. **Subjective:** `powershell tools/verify.ps1`, then *look at* both captures
   (`test_output/verify_close.png`, `verify_wide.png`) next to the previous
   pair. Same-viewpoint comparison is mandatory: the texture-streak bug
   survived four "fixes" because each was validated from a different camera.

## Accessibility / UX baseline

- Every panel control has a hover tooltip explaining what it does and its units.
- `R` regenerates, `F1` toggles the panel; shortcuts never fire while typing.
- UI scale slider (0.75–2×) at the panel footer for readability.
- Expensive actions (regen, export) never block the frame loop; progress is
  always visible; errors surface in red in the panel — never only in the log.

## Performance practices

- Startup generates a 512² preview synchronously, then swaps in the configured
  resolution asynchronously — never make the user stare at a frozen window.
- The stage cache means param iteration costs only the dirty suffix of the
  pipeline. Protect this: any new pipeline stage must join the fingerprint
  cascade and extend `incremental_equals_full`.
- Heavy CPU work (hydrology, meshing, export) belongs on `AsyncComputeTaskPool`.

## For interns: how to add a feature without breaking quality

1. Read the relevant spec in `docs/superpowers/specs/`; write one if the
   feature is non-trivial (decompose first, design second, code third).
2. New simulation code is a pure function on `Heightmap`/fields — no Bevy
   types — so it is testable headless. Write the failing test first.
3. Wire into `pipeline.rs` (new params → new fingerprint member → extend the
   bit-identity test) and `main.rs` (panel slider with tooltip + sensible range).
4. Run the verification loop above. Update the capture reference pair if the
   visual change is intentional, and say so in the commit message.
