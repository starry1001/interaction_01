# Current generation settings

The unit used for matching compression is **pixel area: width × height**. It is not PNG/JPEG bytes or API tokens. Assets are excluded from all budgets and retain their existing processing. `config.py` and its prompt are unchanged.

- **Setting 0**: existing direct generation.
- **Setting 3**: adaptive v2 (`utils/screenshot_delta_adaptive.py`), the formerly preview-only candidate. First screenshot is whole; later screenshots use independent crop bounds, theme-aware stable-layout detection and conservative full-image fallbacks. No cross-step union. The request includes crop coordinates and any coarse palette-change metadata.
- **Setting 1**: compute setting 3 locally on the same source screenshots first. Keep screenshot 0 byte-for-byte; evenly distribute the saved pixel area over the remaining screenshots, then resize each with a uniform width/height scale.
- **Setting 2**: compute the same setting 3 reference. Keep the existing grouping/layout (including 5 → 3 + 2), but construct each sheet from **original-size screenshots** first. Distribute saved area by the number of screenshots in each sheet, then resize each entire sheet uniformly.

## Budget equations

Let original screenshot areas be `P_i`, their sum `P`, adaptive setting 3 output area `T`, and saved area `S = P - T`.

Setting 1 with `n > 1` screenshots:

```text
target_0 = P_0
target_i = P_i - S / (n - 1), i > 0
```

Setting 2 for a sheet containing `k` of the `n` original screenshots:

```text
sheet_target = sum(source screenshot areas in this sheet) - S * k / n
scale = sqrt(sheet_target / raw_sheet_area)
```

`raw_sheet_area` includes label strips and any padding/blank cells. These are included in the output pixel budget: after accounting for the source-area saving, the additional sheet overhead must also be removed. This keeps the actual total close to `T`, rather than accidentally exceeding it through labels/blank space.

For example, if five screenshots save 2,000 pixels under setting 3, setting 1 subtracts 500 from each of the four later source areas. Setting 2 subtracts 1,200 from the three-source sheet's source-area sum and 800 from the two-source sheet's source-area sum; it then resizes the label-inclusive sheets to these targets.

Integer image dimensions introduce a small rounding shortfall; reported actual area is at most the target. Aspect ratio is preserved to integer-pixel rounding. If mixed-size images cannot support the initially assigned saving, the image is capped at one pixel and the remaining saving is redistributed. A single screenshot has no setting 3 saving; setting 1 preserves it whole.

### Additional tile-boundary snapping

Settings 1/2 apply tile-boundary snapping after area-budget resizing; setting 3 now applies it after cropping/full-image fallback as well. `--tile-snap-margin` defaults to **100 pixels**; `0` disables this extra step, and `150` makes it more aggressive. Screenshot 0 stays original in settings 1 and 3; all assets remain untouched. Setting 0 never applies this logic.

Setting 3 records `tile_snap.before_size`, `after_size`, and estimated tile counts in its manifest. Crop bounds always remain in original page coordinates; the API text separately states the actual resized input dimensions. Settings 1/2 compute their reference budget from the **updated setting 3 output including snapping**, with the same margin, so the shared compression baseline remains consistent. The preview script accepts `--tile-snap-margin` too.

For each image, consider a lower width/height boundary only when that dimension is at most the margin above it. Resize both dimensions with the same scale, and accept only if a GPT-5-style high/auto tile estimate actually decreases (including the 2048/768 preprocessing). Among candidates prefer fewer tiles, then the largest image. Repeat until no qualifying boundary remains. The recorded estimates are for this tile-based model family and are not measured API tokens; other model families can use different rules.

For the current setting 1 examples, default 100 leaves sample 1 (`900×443`) and sample 2 (`1314×646`) unchanged. Sample 3's later images become `471×1024` instead of `515×1119`, reducing their estimated tiles from six to two. Margin 150 would also change sample 2.

`compression_budget.json` separates `rounding_shortfall_pixels` from `tile_snap_extra_saved_pixels` and records each image's before/after dimensions and estimated tiles. Consequently settings 1/2 can now retain fewer total pixels than the setting 3 area target, intentionally, to reduce input tokens.

No setting 3 webpage or API call is needed to calculate this reference. Each run computes it directly from the current screenshot inputs, so stale model outputs are not used.

## Records and reruns

Settings 1/2 write `compression_budget.json` in `temp_images`, the model output directory, and the timestamped generation backup. It records source pixels, the setting 3 manifest/target, per-output targets, label/padding overhead (sheets), actual dimensions and rounding difference. Setting 3 still writes `screenshot_manifest.json` and API usage remains in `usage.json`.

```bash
python generate.py --data ./data/data.jsonl --result ./outputs --model gpt-5 --setting 1 --coverage
python generate.py --data ./data/data.jsonl --result ./outputs --model gpt-5 --setting 2 --coverage
python generate.py --data ./data/data.jsonl --result ./outputs --model gpt-5 --setting 3 --coverage
```

Existing outputs are not rewritten automatically; `--coverage` regenerates a selected combination. API token counts may differ even for matched pixel areas because image dimensions, grouping, model preprocessing and setting 3's coordinate/palette text differ. Compare returned `usage.json` for actual token use.

Run only selected task IDs without editing `data.jsonl` (`--ids` is an alias for `--task`):

```bash
python generate.py --data ./data/data.jsonl --result ./outputs --model gpt-5 --setting 1 --task 3 --coverage
# Multiple IDs: --task 1 3
# Restore strict area-budget behavior: --tile-snap-margin 0
```

Unknown IDs are rejected before any tasks run. Filtering affects generation only.

Previews use the same current crop algorithm:

```bash
python preview_setting3.py --task 1 2 3
```

Real-input pixel-budget checks are saved under `outputs/pixel_budget_previews/{1,2,3}/setting{1,2,3}/`; these contain image/metadata artifacts only, not model-generated HTML.
