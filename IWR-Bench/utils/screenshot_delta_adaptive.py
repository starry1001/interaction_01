"""Theme-aware per-step screenshot crops shared by generation and previews."""

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .screenshot_delta import DeltaOptions, analyze_pair as analyze_baseline
from .image_tiles import snap_to_tile_boundary


@dataclass(frozen=True)
class AdaptiveDeltaOptions(DeltaOptions):
    theme_min_ssim: float = 0.90
    theme_min_edge_overlap: float = 0.88
    theme_min_changed_fraction: float = 0.50
    theme_component_pixels: int = 64


def _palette_changes(previous, current, threshold):
    """Summarize color changes at matching positions, without inferring CSS roles."""
    a = previous[::4, ::4, ::-1].reshape(-1, 3)
    b = current[::4, ::4, ::-1].reshape(-1, 3)
    bins = a.astype(np.int32) // 32
    ids = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    counts = np.bincount(ids, minlength=512)
    changes = []
    for key in np.argsort(counts)[::-1][:6]:
        if counts[key] / len(a) < 0.01:
            continue
        selected = ids == key
        before, after = np.median(a[selected], axis=0), np.median(b[selected], axis=0)
        if np.max(np.abs(after - before)) <= threshold:
            continue
        changes.append({'before': '#%02x%02x%02x' % tuple(before.astype(int)),
                        'after': '#%02x%02x%02x' % tuple(after.astype(int)),
                        'coverage_fraction': float(counts[key] / len(a))})
    return changes


def analyze_pair(previous, current, options=AdaptiveDeltaOptions()):
    decision = analyze_baseline(previous, current, options)
    metrics = decision['metrics']
    if (previous.shape != current.shape
            or metrics.get('changed_fraction', 0) < options.theme_min_changed_fraction
            or metrics.get('ssim', 0) < options.theme_min_ssim):
        return decision
    # Compare outline positions with a one-pixel tolerance to theme/antialias changes.
    a, b = [cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 130) > 0
            for image in (previous, current)]
    if min(int(a.sum()), int(b.sum())) < 100:
        return decision
    tolerance = np.ones((3, 3), np.uint8)
    near_a = cv2.dilate(a.astype(np.uint8), tolerance) > 0
    near_b = cv2.dilate(b.astype(np.uint8), tolerance) > 0
    overlap = float(((a & near_b).sum() + (b & near_a).sum()) / (a.sum() + b.sum()))
    metrics['edge_overlap'] = overlap
    if overlap < options.theme_min_edge_overlap:
        return decision
    theme_changes = _palette_changes(previous, current, options.pixel_threshold)
    if not theme_changes:
        return decision
    novel = (a & ~near_b) | (b & ~near_a)
    mask = cv2.dilate(novel.astype(np.uint8), np.ones((5, 5), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= options.theme_component_pixels
    keep[0] = False
    mask = keep[labels].astype(np.uint8)
    metrics['structural_changed_fraction'] = float(mask.mean())
    decision = {'scene_change': False, 'box': None, 'metrics': metrics,
                'theme_changes': theme_changes, 'reason': 'theme_without_reliable_local_edit_keep_full'}
    if not mask.any():
        return decision
    x, y, w, h = cv2.boundingRect(mask)
    height, width = mask.shape
    box = [max(0, x - options.padding), max(0, y - options.padding),
           min(width, x + w + options.padding), min(height, y + h + options.padding)]
    # Include complete text/control rows, not just the digits or tab outlines that changed.
    # This avoids cutting "05:00" after "05:" when only its first digit changed.
    band_columns = np.flatnonzero(np.any((a | b)[box[1]:box[3]], axis=0))
    if len(band_columns):
        box[0] = min(box[0], max(0, int(band_columns[0]) - options.padding))
        box[2] = max(box[2], min(width, int(band_columns[-1]) + 1 + options.padding))
    fraction = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
    metrics['structural_crop_fraction'] = fraction
    if fraction < options.max_crop_fraction:
        decision.update(box=box, reason='theme_change_stable_layout_local_crop')
    else:
        decision['reason'] = 'theme_change_structural_edits_spread_keep_full'
    return decision


def prepare_delta_screenshots(paths, output_dir, options=AdaptiveDeltaOptions(), tile_snap_margin=100):
    """Use per-step boxes without a cross-frame union, preserving the manifest schema."""
    output_dir = Path(output_dir)
    if tile_snap_margin < 0:
        raise ValueError('tile_snap_margin must be nonnegative')
    output_dir.mkdir(parents=True, exist_ok=True)
    records, outputs = [], []
    previous, scene_id = None, 0
    for index, source in enumerate(paths):
        current = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if current is None:
            raise ValueError(f'Could not read screenshot: {source}')
        height, width = current.shape[:2]
        decision = ({'scene_change': True, 'reason': 'first_screenshot', 'box': None, 'metrics': {}}
                    if previous is None else analyze_pair(previous, current, options))
        if index and decision['scene_change']:
            scene_id += 1
        record = {'index': index, 'source': str(source), 'scene_id': scene_id,
                  'original_size': [width, height], 'compared_to': index - 1 if index else None, **decision}
        destination = output_dir / f'screenshot_{index}.png'
        if record['box']:
            with Image.open(source) as image:
                image.crop(tuple(record['box'])).save(destination)
            record['mode'] = 'crop'
        else:
            shutil.copy2(source, destination)
            record.update(mode='full', box=[0, 0, width, height])
        record['tile_snap'] = snap_to_tile_boundary(destination, tile_snap_margin if index else 0)
        with Image.open(destination) as image:
            record['output_size'] = list(image.size)
        record['output'] = destination.name
        record['retained_pixel_fraction'] = record['output_size'][0] * record['output_size'][1] / (width * height)
        records.append(record)
        outputs.append(str(destination))
        previous = current
        print(f"Screenshot {index}: {record['mode']}, box={record['box']}, reason={record['reason']}")
    manifest = {'algorithm': 'theme_aware_per_step_v2', 'preview_only': False,
                'tile_snap_margin': tile_snap_margin,
                'options': asdict(options), 'screenshots': records,
                'limitations': ['Broad recoloring is summarized as palette changes; scene detection is heuristic.',
                                'Color-only edits to existing flat elements may be outside the crop.',
                                'Palette correspondences do not identify exact CSS elements.']}
    (output_dir / 'screenshot_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return outputs
