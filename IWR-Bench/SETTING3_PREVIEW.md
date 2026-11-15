# Experimental setting 3 preview (v2)

> This candidate has now been promoted to production. Current implementation: `utils/screenshot_delta_adaptive.py`; generation and previews share it. Settings 1/2 match its screenshot pixel budget. See [COMPRESSION_SETTINGS.md](COMPRESSION_SETTINGS.md). The remaining text records the earlier preview-only experiment; its statements about production being unchanged are historical.

This candidate is **preview-only**. `generate.py`, `config.py`, `utils/core.py` and the production algorithm in `utils/screenshot_delta.py` are not changed by this experiment. New code lives in `utils/screenshot_delta_preview.py`.

```bash
# Updated gallery for all tasks, with no model/API calls:
python preview_setting3.py

# Inspect selected tasks:
python preview_setting3.py --task 1 2 3

# Generate the production algorithm's results separately for comparison:
python preview_setting3.py --algorithm baseline --output ./outputs/setting3_previews_baseline
```

Open `outputs/setting3_previews/index.html`. This script defaults to `--algorithm experimental`. Running it on a selection rebuilds the index for that selection. Existing unselected task folders remain on disk.

## Changes from production

- **No cross-frame union.** Each screenshot uses only its own change box. A distant cursor in an earlier step no longer enlarges every subsequent crop. If the cursor disappears between the current pair, its disappearance still counts as a change; this candidate does not delete arbitrary small objects or characters as “cursors.”
- **Theme-aware scene check.** When more than 50% of pixels change but grayscale SSIM is at least 0.90, compare Canny edge positions with a one-pixel tolerance. Edge overlap must be at least 0.88, and broad palette changes must also be present. Otherwise use the existing conservative algorithm.
- **Theme-change localization.** For those candidates, use added/removed outlines rather than raw color subtraction. Small fragmented edge differences below 64 dilated pixels are filtered. Add the existing 24-pixel margin, then expand horizontally to include the complete text/control rows intersecting the crop. Keep the full image if the resulting crop occupies at least 65% of the original.
- **Palette changes stay visible.** The manifest records coarse color correspondences and their coverage. The gallery displays before/after swatches below the affected step. The crop still contains original, unmodified current pixels.

## Limits and review points

This is a heuristic for stable layouts undergoing broad recoloring, not a semantic scene recognizer. Color-only edits to existing flat elements can lie outside the crop; palette swatches summarize broad changes but do not specify CSS selectors or exact spatial masks. That loss of spatial color information must be considered before promoting this experiment to generation. The gallery's palette metadata is not sent to any model, because production generation is unchanged.

Frames with insufficient edges, dimensions changing, uncertain structure or widely scattered changes retain the conservative fallback. No screenshots are downsampled or merged. Crop coordinates still refer to the original full image.

Tests:

```bash
python -m unittest discover -s tests -p 'test_screenshot_delta*.py' -v
```
