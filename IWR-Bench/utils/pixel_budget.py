"""Per-image token budgets using the user-selected GPT-5.6 Luna original profile.

Applying this profile to GPT-6 Luna is an explicit experiment assumption, not a
claim that OpenAI has documented identical preprocessing for GPT-6 Luna.
"""

import json
import math
from pathlib import Path

from PIL import Image


PATCH_SIZE = 32
MAX_DIMENSION = 65535
MAX_PATCHES = 30000
RETAIN_NUMERATOR = 3
RETAIN_DENOMINATOR = 5


def patch_count(width, height):
    if width < 1 or height < 1:
        raise ValueError('Image dimensions must be positive')
    return ((width + PATCH_SIZE - 1) // PATCH_SIZE) * ((height + PATCH_SIZE - 1) // PATCH_SIZE)


def estimate_image_tokens(width, height):
    # ceil(patches * 1.2), using exact integer arithmetic.
    return (patch_count(width, height) * 6 + 4) // 5


def original_detail_for_model(model):
    families = ('gpt-6-luna', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra')
    return any(model == family or model.startswith(family + '-') for family in families)


def validate_original_size(width, height):
    if max(width, height) > MAX_DIMENSION:
        raise ValueError(f'Image {width}x{height} exceeds the assumed original-detail dimension limit {MAX_DIMENSION}')
    if patch_count(width, height) > MAX_PATCHES:
        raise ValueError(f'Image {width}x{height} exceeds the assumed original-detail limit of {MAX_PATCHES} patches')


def token_budget_size(width, height, reference_tokens=None):
    """Largest proportional floor-rounded size at or below 60% estimated tokens.

    Token counts are discrete. A one-patch image cannot be reduced below two
    estimated tokens; that exception and original-detail caps are recorded.
    """
    reference = estimate_image_tokens(width, height) if reference_tokens is None else reference_tokens
    if not isinstance(reference, int) or reference < 1:
        raise ValueError('Reference token count must be a positive integer')
    target = reference * RETAIN_NUMERATOR // RETAIN_DENOMINATOR
    effective = min((MAX_PATCHES * 6 + 4) // 5, max(2, target))

    def size_at(scale):
        return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))

    def fits(size):
        w, h = size
        return (max(w, h) <= MAX_DIMENSION and patch_count(w, h) <= MAX_PATCHES
                and estimate_image_tokens(w, h) <= effective)

    lo, hi = 0.0, 1.0
    if fits((width, height)):
        lo = 1.0
    else:
        for _ in range(80):
            mid = (lo + hi) / 2
            if fits(size_at(mid)):
                lo = mid
            else:
                hi = mid
    size = size_at(lo)
    assert fits(size)
    return size, target, effective


def resize_to_token_budget(path, reference_tokens=None):
    with Image.open(path) as image:
        width, height = image.size
        size, target, effective = token_budget_size(width, height, reference_tokens)
        if size != image.size:
            image.resize(size, Image.Resampling.LANCZOS).save(path, compress_level=1)
    source_tokens = estimate_image_tokens(width, height)
    reference = source_tokens if reference_tokens is None else reference_tokens
    output_tokens = estimate_image_tokens(*size)
    return {'source_size': [width, height], 'source_pixels': width * height,
            'source_patches': patch_count(width, height), 'source_estimated_tokens': source_tokens,
            'reference_estimated_tokens': reference,
            'target_estimated_tokens': target, 'effective_target_estimated_tokens': effective,
            'minimum_patch_limited': target < 2,
            'profile_cap_limited': target > effective or max(size) == MAX_DIMENSION,
            'output_size': list(size), 'output_pixels': size[0] * size[1],
            'output_patches': patch_count(*size), 'output_estimated_tokens': output_tokens,
            'retained_token_fraction': output_tokens / reference,
            'retained_vs_unscaled_image_fraction': output_tokens / source_tokens,
            'rounding_shortfall_tokens': effective - output_tokens,
            'output': Path(path).name}


def write_budget(allocations, output_dir, setting):
    source_tokens = sum(r['source_estimated_tokens'] for r in allocations)
    reference_tokens = sum(r['reference_estimated_tokens'] for r in allocations)
    output_tokens = sum(r['output_estimated_tokens'] for r in allocations)
    budget = {'setting': setting, 'unit': 'estimated image input tokens',
              'profile': {'reference_model': 'gpt-5.6-luna', 'assumed_for': 'gpt-6-luna',
                          'assumption_not_officially_verified': True, 'detail': 'original',
                          'patch_size': PATCH_SIZE, 'token_multiplier': 1.2,
                          'max_dimension': MAX_DIMENSION, 'max_patches': MAX_PATCHES},
              'target_retained_fraction': 0.6,
              'baseline': 'each original screenshot' if setting == 1 else 'sum of original screenshot token estimates in each group, excluding added labels and padding',
              'reference_estimated_tokens': reference_tokens,
              'source_estimated_tokens': source_tokens, 'output_estimated_tokens': output_tokens,
              'retained_token_fraction': output_tokens / reference_tokens if reference_tokens else 0,
              'retained_vs_unscaled_image_fraction': output_tokens / source_tokens if source_tokens else 0,
              'source_pixels': sum(r['source_pixels'] for r in allocations),
              'output_pixels': sum(r['output_pixels'] for r in allocations),
              'allocations': allocations}
    (Path(output_dir) / 'compression_budget.json').write_text(
        json.dumps(budget, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Screenshot token estimate: reference={reference_tokens}, unscaled input={source_tokens}, output={output_tokens}, retained={budget["retained_token_fraction"]:.2%}')
    return budget
