# Setting 3: full scene anchors and change crops

> Current behavior: adaptive v2 has been promoted to generation; settings 1/2 now match its screenshot pixel budget. See [COMPRESSION_SETTINGS.md](COMPRESSION_SETTINGS.md) for the current rules. The algorithm thresholds and scene union described below document the **legacy baseline**, still available using `preview_setting3.py --algorithm baseline`.

Run generation with `--setting 3`. Assets remain first; the first webpage screenshot is full resolution. Later screenshots are either full images or unscaled crops in chronological order. Settings 0–2 and the prompt in `config.py` are unchanged.

```bash
python generate.py --data ./data/data.jsonl --result ./outputs --model gpt-6-luna --setting 3
```

## Algorithm

1. Compare each screenshot with the **previous original screenshot**, not its cropped output.
2. Different dimensions → keep the current image whole and start a new scene.
3. Apply a 3×3 Gaussian blur, then threshold maximum per-channel absolute differences at 18/255. Discard connected components smaller than 8 pixels. Do not erode: thin progress bars and text edits must survive.
4. Compute grayscale local SSIM at up to 960 pixels on the longest side. More than 35% changed pixels or SSIM below 0.55 → keep the current image whole and start a new scene. This is a conservative visual heuristic, not semantic scene recognition.
5. Close small gaps in the difference mask, take a bounding rectangle, and add 24 pixels of context. If it occupies at least 65% of the page, keep the whole image. If no reliable change is found, also keep the whole image rather than dropping a state.
6. For localized changes within a scene, use the union of candidate crop boxes when that union still occupies less than 65% of the page. This provides stable context such as a game board and score together. Otherwise retain each individual candidate crop. Every crop comes from the current original image without resizing.
7. Add text before each screenshot in the API request: screenshot index, original page dimensions, full/crop status, crop bounds, and the comparison predecessor. Coordinates are original-image pixels with exclusive right/bottom edges. Crops are explicitly identified as regions of an existing page, not separate pages.

Thresholds live in `DeltaOptions` in `utils/screenshot_delta.py`. These are engineering defaults, validated on synthetic cases and the two example tasks, not a benchmark-trained optimum. Lower the pixel threshold/minimum component size to preserve finer changes; reduce the scene/crop thresholds to fall back to full images more often.

## Inspect without calling a model

The standalone script builds an HTML gallery, annotated original thumbnails, processed images, and a `preview.png` per task. By default it processes all tasks under `data/data_backup`:

The gallery now defaults to a **preview-only experimental v2** (theme-aware, per-step crop bounds). Production `--setting 3` still uses the algorithm described above. Use `--algorithm baseline` to preview the unchanged production algorithm. See [SETTING3_PREVIEW.md](SETTING3_PREVIEW.md).

```bash
python preview_setting3.py
# Or selected tasks / another dataset root:
python preview_setting3.py --task 1 2
python preview_setting3.py --data-root ./data/data --output ./outputs/setting3_previews
```

Open `outputs/setting3_previews/index.html`. The left image shows the original and crop rectangle; the right image shows the candidate algorithm output. Click either image to inspect at original size. `summary.json` records full/crop counts and screenshot pixel reductions. `--pixel-threshold` and `--padding` allow preview experiments without editing generation defaults. A selected-task run rebuilds the index for that selection only; other task directories remain on disk.

To produce only processed screenshots and the manifest for one directory:

```bash
python -m utils.screenshot_delta \
  --checkpoints ./data/data_backup/2/checkpoints \
  --output ./outputs/setting3_previews/2/temp_images
```

`screenshot_manifest.json` records the source, comparison predecessor, scene ID, full/crop decision, reason, metrics, bounds, dimensions and retained pixel fraction for every image. Generation copies it into the model output directory and each timestamped backup.

Previews of the two supplied tasks are under `outputs/setting3_previews/1/preview.png` and `outputs/setting3_previews/2/preview.png`. These are inspection artifacts, not model-generated webpages.

## Limitations

- Pixel differences cannot identify which element the user intended to interact with. Animations, artwork and clock changes remain visible changes and may enlarge crops.
- Small/low-contrast changes below the configured tolerance can be omitted. The metadata explicitly describes that tolerance; identical or inconclusive frames are retained whole.
- Scrolling, page reflow and widely dispersed edits may fall back to full frames. This version deliberately does not use row/column Git diff or image warping, which can hide layout movement or require additional alignment assumptions.
- Saving pixels/bytes does not imply the same proportional token savings. Compare actual `usage.json`; coordinate labels add some input text tokens.

## References

- [Interaction2Code §III-C / Algorithm 1](https://arxiv.org/pdf/2411.03292): pixel-difference bounding boxes for equal-size images, row/column diff for unequal-size images. Setting 3 extends the equal-size approach and keeps unequal-size screenshots whole.
- [OpenCV connected components](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html) and [morphological transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html): binary-mask filtering and gap closing.
- [SSIM overview](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html): local structural comparison, used here as one scene-change signal rather than the only criterion.

## Validation

```bash
python -m unittest discover -s tests -p test_screenshot_delta.py -v
```
