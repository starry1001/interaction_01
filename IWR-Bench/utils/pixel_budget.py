"""Match screenshot pixel areas across generation settings (not API token counts)."""

import json
import math
import tempfile
from pathlib import Path

from PIL import Image

from .screenshot_delta_adaptive import prepare_delta_screenshots
from .image_tiles import tile_count, snap_to_tile_boundary


def reference_budget(paths, output_dir, tile_snap_margin=100):
    with tempfile.TemporaryDirectory(prefix='delta_reference_', dir=output_dir) as folder:
        prepare_delta_screenshots(paths, folder, tile_snap_margin=tile_snap_margin)
        manifest = json.loads((Path(folder) / 'screenshot_manifest.json').read_text(encoding='utf-8'))
    original = sum(r['original_size'][0] * r['original_size'][1] for r in manifest['screenshots'])
    retained = sum(r['output_size'][0] * r['output_size'][1] for r in manifest['screenshots'])
    return {'unit': 'pixels (width * height)', 'source_screenshot_pixels': original,
            'setting3_target_pixels': retained, 'saved_pixels': original - retained,
            'setting3_reference': manifest}


def allocate_targets(areas, saving, weights=None):
    """Share savings by weight, redistributing at the one-pixel-per-image floor."""
    weights = weights or [1] * len(areas)
    targets = [float(area) for area in areas]
    remaining = float(saving)
    active = set(range(len(areas)))
    while remaining > 1e-6 and active:
        weight = sum(weights[i] for i in active)
        capped = [i for i in active if remaining * weights[i] / weight >= targets[i] - 1]
        if not capped:
            for i in active:
                targets[i] -= remaining * weights[i] / weight
            remaining = 0
        else:
            for i in capped:
                remaining -= targets[i] - 1
                targets[i] = 1.0
                active.remove(i)
    if remaining > 1e-5:
        raise ValueError('Cannot fit pixel budget while retaining at least one pixel per output image')
    return targets


def resize_to_area(path, target_pixels):
    with Image.open(path) as image:
        width, height = image.size
        target = max(1, min(width * height, math.floor(target_pixels + 1e-6)))
        if target == width * height:
            return [width, height]
        scale = math.sqrt(target / (width * height))
        new_width = max(1, math.floor(width * scale))
        new_height = max(1, math.floor(height * scale))
        if new_width * new_height > target:
            if new_width >= new_height:
                new_width = max(1, target // new_height)
            else:
                new_height = max(1, target // new_width)
        image.resize((new_width, new_height), Image.Resampling.LANCZOS).save(path)
    return [new_width, new_height]


def finish_budget(budget, paths, allocations, output_dir, setting):
    achieved = 0
    for path, record in zip(paths, allocations):
        with Image.open(path) as image:
            record['output_size'] = list(image.size)
            record['actual_pixels'] = image.width * image.height
        record['output'] = Path(path).name
        achieved += record['actual_pixels']
    snap_saving = sum(r.get('tile_snap', {}).get('extra_saved_pixels', 0) for r in allocations)
    budget.update(setting=setting, allocations=allocations, actual_screenshot_pixels=achieved,
                  tile_snap_extra_saved_pixels=snap_saving,
                  rounding_shortfall_pixels=budget['setting3_target_pixels'] - achieved - snap_saving,
                  total_shortfall_pixels=budget['setting3_target_pixels'] - achieved)
    (Path(output_dir) / 'compression_budget.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Screenshot pixel budget: setting3={budget['setting3_target_pixels']}, actual={achieved}, rounding={budget['rounding_shortfall_pixels']}, extra tile snapping={snap_saving}")
