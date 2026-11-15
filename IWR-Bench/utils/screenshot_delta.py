"""Conservative screenshot deltas for generation setting 3 (no model/API calls)."""

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DeltaOptions:
    pixel_threshold: int = 18
    min_component_pixels: int = 8
    padding: int = 24
    max_changed_fraction: float = 0.35
    min_scene_ssim: float = 0.55
    max_crop_fraction: float = 0.65


def _ssim(previous, current):
    """Mean local SSIM on a reduced grayscale image; scene gate, not ROI detector."""
    height, width = previous.shape[:2]
    scale = min(1.0, 960 / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    a, b = [cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size).astype(np.float32)
            for image in (previous, current)]
    mean_a, mean_b = [cv2.GaussianBlur(image, (11, 11), 1.5) for image in (a, b)]
    variance_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mean_a * mean_a
    variance_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mean_b * mean_b
    covariance = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mean_a * mean_b
    score = ((2 * mean_a * mean_b + 6.5025) * (2 * covariance + 58.5225)) / (
        (mean_a * mean_a + mean_b * mean_b + 6.5025) * (variance_a + variance_b + 58.5225)
    )
    return float(np.clip(score.mean(), -1, 1))


def analyze_pair(previous, current, options=DeltaOptions()):
    """Return a conservative change box in CURRENT image coordinates (exclusive right/bottom)."""
    if previous.shape != current.shape:
        return {'scene_change': True, 'reason': 'dimensions_changed', 'box': None, 'metrics': {}}
    height, width = current.shape[:2]
    a, b = [cv2.GaussianBlur(image, (3, 3), 0) for image in (previous, current)]
    difference = cv2.absdiff(a, b).max(axis=2)
    mask = (difference > options.pixel_threshold).astype(np.uint8)
    # Do not erode the mask: small text edits and thin progress bars matter.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= options.min_component_pixels
    keep[0] = False
    filtered = keep[labels].astype(np.uint8)
    changed = float(filtered.mean())
    structural_similarity = _ssim(previous, current)
    metrics = {'changed_fraction': changed, 'ssim': structural_similarity}
    if changed > options.max_changed_fraction or structural_similarity < options.min_scene_ssim:
        return {'scene_change': True, 'reason': 'large_visual_change', 'box': None, 'metrics': metrics}
    if not filtered.any():
        return {'scene_change': False, 'reason': 'no_reliable_change_keep_full', 'box': None, 'metrics': metrics}
    # Closing joins nearby character strokes without dropping isolated meaningful edits.
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    x, y, w, h = cv2.boundingRect(filtered)
    box = [max(0, x - options.padding), max(0, y - options.padding),
           min(width, x + w + options.padding), min(height, y + h + options.padding)]
    area = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
    metrics['candidate_crop_fraction'] = area
    if area >= options.max_crop_fraction:
        return {'scene_change': False, 'reason': 'changes_spread_keep_full', 'box': None, 'metrics': metrics}
    return {'scene_change': False, 'reason': 'localized_change', 'box': box, 'metrics': metrics}


def prepare_delta_screenshots(paths, output_dir, options=DeltaOptions()):
    """Keep scene anchors whole and crop subsequent screenshots without resizing them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    previous = None
    scene_id = 0
    for index, source in enumerate(paths):
        current = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if current is None:
            raise ValueError(f'Could not read screenshot: {source}')
        height, width = current.shape[:2]
        decision = ({'scene_change': True, 'reason': 'first_screenshot', 'box': None, 'metrics': {}}
                    if previous is None else analyze_pair(previous, current, options))
        if index and decision['scene_change']:
            scene_id += 1
        records.append({'index': index, 'source': str(source), 'scene_id': scene_id,
                        'original_size': [width, height], 'compared_to': index - 1 if index else None,
                        **decision})
        previous = current

    # A stable ROI within a scene retains context (e.g. the entire changing game board + score).
    # Only cropped frames participate; full frames are always preserved as-is.
    for scene in sorted({record['scene_id'] for record in records}):
        members = [record for record in records if record['scene_id'] == scene and record['box']]
        if not members:
            continue
        boxes = [record['box'] for record in members]
        union = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                 max(b[2] for b in boxes), max(b[3] for b in boxes)]
        width, height = members[0]['original_size']
        if (union[2] - union[0]) * (union[3] - union[1]) / (width * height) < options.max_crop_fraction:
            for record in members:
                record['box'] = union.copy()
                record['reason'] = 'localized_change_scene_union'

    outputs = []
    for record in records:
        destination = output_dir / f"screenshot_{record['index']}.png"
        box = record['box']
        if box:
            with Image.open(record['source']) as image:
                image.crop(tuple(box)).save(destination)
            record['mode'] = 'crop'
        else:
            shutil.copy2(record['source'], destination)
            record['mode'] = 'full'
            width, height = record['original_size']
            record['box'] = [0, 0, width, height]
        record['output'] = destination.name
        with Image.open(destination) as image:
            record['output_size'] = list(image.size)
        width, height = record['original_size']
        record['retained_pixel_fraction'] = record['output_size'][0] * record['output_size'][1] / (width * height)
        outputs.append(str(destination))
        print(f"Screenshot {record['index']}: {record['mode']}, box={record['box']}, reason={record['reason']}")
    manifest = {'algorithm': 'thresholded_color_difference_components_ssim_scene_gate_v1',
                'options': asdict(options), 'screenshots': records}
    (output_dir / 'screenshot_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return outputs


if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser(description='Preview setting 3 crops without calling a model.')
    parser.add_argument('--checkpoints', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    ordered = sorted((int(match.group(1)), str(path)) for path in args.checkpoints.glob('step_*.png')
                     if (match := re.fullmatch(r'step_(\d+)\.png', path.name)))
    if not ordered:
        parser.error('No step_<number>.png screenshots found')
    if args.checkpoints.resolve() == args.output.resolve():
        parser.error('Output must differ from the source screenshot directory')
    prepare_delta_screenshots([path for _, path in ordered], args.output)
