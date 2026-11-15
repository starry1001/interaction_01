"""Shared tile-aware image resizing, independent of screenshot scene analysis."""

import math
from PIL import Image


def tile_count(width, height):
    """GPT-5 high/auto-style estimate, not an API measurement."""
    scale = min(1.0, 2048 / max(width, height))
    width, height = width * scale, height * scale
    scale = min(1.0, 768 / min(width, height))
    width, height = max(1, math.floor(width * scale)), max(1, math.floor(height * scale))
    return math.ceil(width / 512) * math.ceil(height / 512)


def snap_to_tile_boundary(path, margin=100):
    """Downsize near 512px boundaries only if the estimated tile count falls."""
    with Image.open(path) as image:
        before = current = image.size
        while margin > 0:
            width, height = current
            candidates = []
            for axis, length in enumerate(current):
                boundary = ((length - 1) // 512) * 512
                if boundary <= 0 or length - boundary > margin:
                    continue
                scale = boundary / length
                size = ((boundary, max(1, math.floor(height * scale))) if axis == 0
                        else (max(1, math.floor(width * scale)), boundary))
                if tile_count(*size) < tile_count(*current):
                    candidates.append(size)
            if not candidates:
                break
            current = min(candidates, key=lambda size: (tile_count(*size), -size[0] * size[1]))
        if current != before:
            image.resize(current, Image.Resampling.LANCZOS).save(path)
    return {'margin': margin, 'before_size': list(before), 'after_size': list(current),
            'estimated_tiles_before': tile_count(*before), 'estimated_tiles_after': tile_count(*current),
            'extra_saved_pixels': before[0] * before[1] - current[0] * current[1]}


