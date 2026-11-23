"""Shared Setting 3 screenshot crops for preview, generation, and pixel budgets.

Coordinates refer to the original viewport. Matching is approximate, not lossless.
"""

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
    max_crop_fraction: float = 0.65
    theme_min_ssim: float = 0.90
    theme_min_edge_overlap: float = 0.88
    theme_min_changed_fraction: float = 0.50
    theme_component_pixels: int = 64


ALGORITHM = 'scroll_single_crop_v3'


def _ssim(previous, current):
    """Mean local SSIM on a reduced grayscale image for stable-layout theme checks."""
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


def _analyze_theme(previous, current, options):
    a, b = [cv2.GaussianBlur(image, (3, 3), 0) for image in (previous, current)]
    mask = (_difference(a, b) > options.pixel_threshold).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= options.min_component_pixels
    keep[0] = False
    metrics = {'changed_fraction': float(keep[labels].mean()), 'ssim': _ssim(previous, current)}
    decision = {}
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


def _difference(a, b):
    return cv2.absdiff(a, b).max(axis=2)


def detect_scroll(previous, current, threshold=18):
    """Find vertical translation from feature votes, then verify actual overlap pixels.

    dy is current_y - previous_y: negative means scrolling down. Stationary headers
    and sidebars may coexist with the translated content; they vote for zero instead.
    """
    h, w = current.shape[:2]
    scale = min(1., 1280 / max(h, w))
    size = (round(w * scale), round(h * scale))
    a, b = [cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), size)
            for im in (previous, current)]
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=12)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or min(len(da), len(db)) < 12:
        return None
    votes = []
    for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2):
        if len(pair) != 2 or pair[0].distance >= .72 * pair[1].distance:
            continue
        m = pair[0]
        x, y = ka[m.queryIdx].pt
        xx, yy = kb[m.trainIdx].pt
        if abs(xx - x) <= 2.5 and 24 * scale < abs(yy - y) < .88 * a.shape[0]:
            votes.append((x / scale, y / scale, (yy - y) / scale))
    if len(votes) < 10:
        return None
    votes = np.asarray(votes)
    bins, counts = np.unique(np.rint(votes[:, 2] / 4).astype(int), return_counts=True)
    for key in bins[np.argsort(counts)[::-1][:4]]:
        support = votes[np.abs(votes[:, 2] - key * 4) <= 4]
        if len(support) < 10 or np.ptp(support[:, 0]) < w * .12:
            continue
        dy = int(round(np.median(support[:, 2])))
        height = h - abs(dy)
        if height < h * .12 or np.ptp(support[:, 1]) < min(height * .25, h * .12):
            continue
        # Integer refinement avoids resampling content and verifies feature proposals.
        best = None
        left = max(0, int(support[:, 0].min()) - 16)
        right = min(w, int(support[:, 0].max()) + 17)
        for shift in range(dy - 2, dy + 3):
            p, c = max(0, -shift), max(0, shift)
            ah, bh = previous[p:p + h - abs(shift)], current[c:c + h - abs(shift)]
            ah, bh = ah[:, left:right], bh[:, left:right]
            edges = cv2.Canny(cv2.cvtColor(ah, cv2.COLOR_BGR2GRAY), 50, 130) > 0
            diff = _difference(cv2.GaussianBlur(ah, (3, 3), 0), cv2.GaussianBlur(bh, (3, 3), 0))
            good = edges & (diff <= threshold)
            score = float(good.sum() / max(1, edges.sum()))
            if best is None or score > best[0]:
                best = score, shift, int(good.sum())
        score, dy, matched_edges = best
        if score >= .60 and matched_edges >= 150:
            return {'dy': dy, 'direction': 'down' if dy < 0 else 'up',
                    'matched_features': len(support), 'edge_pixel_agreement': score,
                    'overlap_height': h - abs(dy)}
    return None


def _expand_visible_components(mask, current):
    """Keep complete connected text lines/images touched by a detected edit.

    This uses visual connectivity, not semantic DOM components. In particular it
    avoids producing a few isolated letters when only a substring has changed.
    """
    edges = cv2.Canny(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), 50, 130)
    lines = cv2.dilate(edges, np.ones((5, 31), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    touches = np.bincount(labels[mask > 0], minlength=len(stats))
    touched = np.flatnonzero(touches)
    expanded = mask.copy()
    h, w = mask.shape
    for label in touched:
        if not label:
            continue
        x, y, width, height, area = stats[label]
        if (area < 8 or (height <= 8 and width > .8 * w)
                or (width < .03 * w and height > .6 * h)
                or (width > .75 * w and height > .75 * h and area < width * height * .1)
                or (width * height > .12 * w * h and touches[label] < .005 * width * height)):
            continue
        expanded[int(y):min(h, int(y + height)), int(x):min(w, int(x + width))] = 1
    return expanded


def _scrollbar_width(previous, current):
    """Recognize a thin neutral track and smooth thumb on light or dark pages.

    Inspect a bounded edge strip, infer the thumb's actual left edge, and require
    evidence in both frames. Colored controls and text are not removed wholesale.
    """
    h, w = current.shape[:2]
    width = min(32, max(16, w // 20))
    left_edges = []
    for image in (previous, current):
        strip = image[:, -width:]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        neutral = (strip.max(axis=2) - strip.min(axis=2) <= 8).mean(axis=0)
        good = []
        for x, column in enumerate(gray.T):
            background = float(np.median(column))
            if neutral[x] < .99 or not (background >= 220 or background <= 60):
                continue
            thumb = np.abs(column.astype(float) - background) >= 20
            transitions = np.diff(np.r_[False, thumb, False].astype(np.int8))
            lengths = np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)
            if (len(lengths) and max(12, h * .06) <= lengths.max() <= h * .6
                    and (np.abs(np.diff(column.astype(int))) > 18).sum() <= 18):
                good.append(x)
        runs = np.split(np.asarray(good, dtype=int), np.flatnonzero(np.diff(good) != 1) + 1)
        runs = [run for run in runs if len(run) >= 3]
        if not runs:
            return 0
        left = max(0, int(min(run[0] for run in runs)) - 2)
        if neutral[left:].mean() < .99:
            return 0
        left_edges.append(left)
    return width - min(left_edges)


def _snap_rows(boxes, current):
    """Move crop boundaries outward to nearby empty rows to avoid sliced text."""
    edges = cv2.Canny(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), 50, 130) > 0
    h = current.shape[0]
    result = []
    for x1, y1, x2, y2 in boxes:
        occupied = edges[:, x1:x2].any(axis=1)
        if y1 and occupied[y1]:
            for y in range(y1, max(-1, y1 - 49), -1):
                if y >= 2 and not occupied[y - 2:y + 1].any():
                    y1 = y
                    break
        if y2 < h and occupied[y2 - 1]:
            for y in range(y2, min(h, y2 + 49)):
                if not occupied[y:min(h, y + 3)].any():
                    y2 = y
                    break
        result.append([x1, y1, x2, y2])
    return result


def _trim_blank_sides(box, previous, current, scrollbar, padding):
    """Trim continuous side backgrounds, protecting content in BOTH frames.

    A deleted element must not be mistaken for an empty margin. Background is
    estimated per row from matching, flat outer edges; uncertain rows prevent trim.
    """
    x1, y1, x2, y2 = box
    limit = current.shape[1] - scrollbar
    foreground = np.zeros(limit, dtype=bool)
    for image in (previous, current):
        band = image[y1:y2, :limit].astype(np.int16)
        if band.shape[1] < 12:
            return box
        left = np.median(band[:, :3], axis=1).astype(np.int16)
        right = np.median(band[:, -3:], axis=1).astype(np.int16)
        flat = ((np.abs(left - right).max(axis=1) <= 8)
                & (np.abs(band[:, :3] - left[:, None]).max(axis=(1, 2)) <= 8)
                & (np.abs(band[:, -3:] - right[:, None]).max(axis=(1, 2)) <= 8))
        if not flat.all():
            return box
        different = (np.abs(band - ((left + right) // 2)[:, None]).max(axis=2) > 14)
        foreground |= different.any(axis=0)
    columns = np.flatnonzero(foreground)
    if not len(columns):
        return box
    left = max(x1, max(0, int(columns[0]) - padding))
    right = min(x2, limit, int(columns[-1]) + 1 + padding)
    return [left, y1, right, y2] if left < right else box


def analyze_pair(previous, current, options=DeltaOptions()):
    h, w = current.shape[:2]
    full = [0, 0, w, h]
    if previous.shape != current.shape:
        return {'boxes': [full], 'reason': 'dimensions_changed_anchor', 'metrics': {}, 'scroll': None}
    # Browser chrome is independent of whether content translation can be verified.
    scrollbar = _scrollbar_width(previous, current)
    scroll = detect_scroll(previous, current, options.pixel_threshold)
    a, b = [cv2.GaussianBlur(im, (3, 3), 0) for im in (previous, current)]
    identity = _difference(a, b)
    # Reuse the existing specialized theme detector, but never its broad scene gate.
    theme = (_analyze_theme(previous, current, options)
             if scroll is None and (identity > options.pixel_threshold).mean() >= options.theme_min_changed_fraction else {})
    if theme.get('theme_changes') and theme.get('box'):
        box = _trim_blank_sides(theme['box'], previous, current, scrollbar, options.padding)
        return {'boxes': [box], 'reason': 'theme_and_local_edit',
                'theme_changes': theme['theme_changes'], 'scroll': None,
                'metrics': theme['metrics']}
    residual = identity.copy()
    if scroll:
        dy = scroll['dy']
        py, cy = max(0, -dy), max(0, dy)
        height = h - abs(dy)
        pa, cb = previous[py:py + height], current[cy:cy + height]
        shifted = _difference(cv2.GaussianBlur(pa, (7, 7), 0), cv2.GaussianBlur(cb, (7, 7), 0))
        # Video-derived frames have subpixel text rasterization differences. Compare
        # stable outlines as well as low-frequency color; genuine new edges remain.
        ea, eb = [cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (3, 3), 0), 50, 130) > 0 for im in (pa, cb)]
        kernel = np.ones((5, 5), np.uint8)
        novel_edges = (ea & ~(cv2.dilate(eb.astype(np.uint8), kernel) > 0)) | (eb & ~(cv2.dilate(ea.astype(np.uint8), kernel) > 0))
        shifted[novel_edges] = 255
        # Reuse a block only when most pixels match one coherent correspondence.
        # This avoids mixing arbitrary per-pixel matches from two different origins.
        tile = 32
        for y in range(0, height, tile):
            for x in range(0, w, tile):
                local = shifted[y:y + tile, x:x + tile]
                if (local <= options.pixel_threshold).mean() >= .90:
                    residual[cy + y:cy + y + local.shape[0], x:x + local.shape[1]] = np.minimum(
                        residual[cy + y:cy + y + local.shape[0], x:x + local.shape[1]], local)
    mask = (residual > options.pixel_threshold).astype(np.uint8)
    if scrollbar:
        mask[:, w - scrollbar:] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= options.min_component_pixels
    keep[0] = False
    mask = keep[labels].astype(np.uint8)
    metrics = {'novel_pixel_fraction': float(mask.mean()),
               'ignored_scrollbar_width': scrollbar,
               'identity_changed_fraction': float((identity > options.pixel_threshold).mean())}
    if not mask.any():
        return {'boxes': [full], 'reason': 'no_reliable_change_anchor', 'metrics': metrics, 'scroll': scroll}
    # Padding supplies unchanged context, joins character strokes, and protects rows
    # at the old/new overlap boundary. Never remove content from the first screenshot.
    padding = max(options.padding, 32 if scroll else 0)
    mask = _expand_visible_components(mask, current)
    mask = cv2.dilate(mask, np.ones((2 * padding + 1, 2 * padding + 1), np.uint8))
    if scrollbar:
        mask[:, w - scrollbar:] = 0
    # Preserve the spatial relationship between all edits in one enclosing crop.
    x, y, width, height = cv2.boundingRect(mask)
    boxes = _snap_rows([[x, y, x + width, y + height]], current)
    box = _trim_blank_sides(boxes[0], previous, current, scrollbar, padding)
    metrics['side_margin_trim'] = {'left': box[0] - boxes[0][0], 'right': boxes[0][2] - box[2]}
    boxes = [box]
    metrics['crop_area_fraction'] = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
    return {'boxes': boxes, 'reason': 'scroll_increment' if scroll else 'spatial_increment',
            'metrics': metrics, 'scroll': scroll}


def prepare_delta_screenshots(paths, output_dir, options=DeltaOptions()):
    """Write one unscaled source crop per frame, with context stored in the manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, outputs = [], []
    previous, scene_id = None, 0
    for index, source in enumerate(paths):
        current = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if current is None:
            raise ValueError(f'Could not read screenshot: {source}')
        h, w = current.shape[:2]
        full = [0, 0, w, h]
        decision = ({'boxes': [full], 'reason': 'first_screenshot', 'metrics': {}, 'scroll': None}
                    if previous is None else analyze_pair(previous, current, options))
        scroll = decision['scroll']
        note = (f"Scroll {scroll['direction']} by {abs(scroll['dy'])} px; reference screenshot_{index - 1}"
                if scroll else f'Reference screenshot_{index - 1}; unchanged regions omitted')
        destination = output_dir / f'screenshot_{index}.png'
        boxes = decision['boxes']
        box = boxes[0]
        if box == full:
            shutil.copy2(source, destination)
            mode, size = 'full', [w, h]
        else:
            with Image.fromarray(cv2.cvtColor(current, cv2.COLOR_BGR2RGB)) as image:
                cropped = image.crop(box)
                cropped.save(destination, compress_level=1)
                mode, size = 'crop', list(cropped.size)
        if index and mode == 'full' and scroll is None:
            scene_id += 1
        record = {'index': index, 'source': str(source), 'original_size': [w, h],
                  'compared_to': index - 1 if index else None, 'scene_id': scene_id,
                  **decision, 'boxes': boxes, 'box': box,
                  'mode': mode, 'output': destination.name, 'output_size': size,
                  'retained_pixel_fraction': size[0] * size[1] / (w * h),
                  'context_note': note if index else 'Full original viewport anchor'}
        records.append(record)
        outputs.append(str(destination))
        previous = current
        print(f"Screenshot {index}: {mode}, boxes={boxes}, reason={record['reason']}")
    manifest = {'algorithm': ALGORITHM, 'preview_only': False,
                'options': asdict(options), 'screenshots': records,
                'limitations': ['Approximate pixel matching; no DOM or semantic component boundaries.',
                               'Vertical scrolling only; ambiguous correspondences retain more content.',
                               'Crops require original viewport coordinates and frame references.',
                               'Theme color correspondences do not identify exact CSS elements.',
                               'Crops retain their original pixels without resizing.']}
    (output_dir / 'screenshot_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return outputs
