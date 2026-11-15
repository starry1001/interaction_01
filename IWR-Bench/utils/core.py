import os
import cv2
import base64
import shutil
import json
import re
from typing import List, Tuple
from config import API_CONFIG, PROMPT_TEMPLATE
from openai import OpenAI
import httpx
from PIL import Image, ImageDraw, ImageFont
from .screenshot_delta_adaptive import prepare_delta_screenshots
from .pixel_budget import reference_budget, allocate_targets, resize_to_area, finish_budget, snap_to_tile_boundary


def load_examples(jsonl_path: str):
	entries = []
	with open(jsonl_path, 'r', encoding='utf-8') as f:
		for line in f:
			if not line.strip():
				continue
			try:
				j = json.loads(line)
				entries.append(j)
			except Exception:
				continue
	return entries

def get_config(model_name: str):
    # Return best-matching config if present, else default
    for key in API_CONFIG:
        if key in model_name.lower():
            return API_CONFIG[key]
    return API_CONFIG["default"]


def encode_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def save_base64_image(img_b64: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(img_b64))

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
             '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webp': 'image/webp', '.svg': 'image/svg+xml'}
    return types.get(ext, 'application/octet-stream')


def resize_image(img_path: str, max_size: int):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: cannot read image {img_path}")
        return

    h, w = img.shape[:2]
    if w > h:
        ratio = max_size / w
        new_w = max_size
        new_h = int(h * ratio)
    else:
        ratio = max_size / h
        new_h = max_size
        new_w = int(w * ratio)

    if ratio < 1.0:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, resized)


def extract_frames(video_path: str, output_folder: str, config_dict: dict) -> List[str]:
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    video_duration = total_frames / fps
    video_fps = config_dict.get('video_fps', 1)
    max_video_frames = config_dict.get('max_video_frames', 64)

    if video_duration > max_video_frames:
        frame_interval = total_frames / max_video_frames
        target_frame_count = max_video_frames
    else:
        frame_interval = fps / video_fps
        target_frame_count = int(video_duration * video_fps)

    frame_paths = []
    for i in range(target_frame_count):
        frame_index = int(i * frame_interval)
        if frame_index >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

    cap.release()
    return frame_paths


def find_files(project_path: str) -> Tuple[List[str], List[str]]:
    """Find stitched assets and screenshots in numeric step order; no video is required."""
    image_paths = []
    stitched_dir = os.path.join(project_path, 'stitched_assets')
    if os.path.exists(stitched_dir):
        for fname in os.listdir(stitched_dir):
            if fname.startswith('stitched_image'):
                image_paths.append(os.path.join(stitched_dir, fname))
    checkpoint_dir = os.path.join(project_path, 'checkpoints')
    checkpoint_paths = []
    if os.path.isdir(checkpoint_dir):
        for fname in os.listdir(checkpoint_dir):
            match = re.fullmatch(r'step_(\d+)\.png', fname)
            if match and os.path.isfile(os.path.join(checkpoint_dir, fname)):
                checkpoint_paths.append((int(match.group(1)), os.path.join(checkpoint_dir, fname)))
    natural_key = lambda path: [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', path)]
    return sorted(image_paths, key=natural_key), [path for _, path in sorted(checkpoint_paths)]


def get_resolution(project_path: str) -> str:
    metadata_path = os.path.join(project_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            return meta.get('recording_details', {}).get('resolution', '1920x1080')
    return '1920x1080'


def screenshot_group_sizes(count: int) -> List[int]:
    """Balance groups of at most six, splitting each five-image group into 3 + 2."""
    if count <= 0:
        return []
    group_count = (count + 5) // 6
    group_size, extra = divmod(count, group_count)
    sizes = []
    for index in range(group_count):
        size = group_size + (index < extra)
        sizes.extend([3, 2] if size == 5 else [size])
    return sizes


def merge_screenshots(checkpoint_paths: List[str], temp_folder: str, budget: dict | None = None, tile_snap_margin: int = 100) -> List[str]:
    """Split ordered screenshots into balanced groups of at most six and label every tile."""
    os.makedirs(temp_folder, exist_ok=True)
    count = len(checkpoint_paths)
    if not count:
        return []
    group_sizes = screenshot_group_sizes(count)
    font = ImageFont.load_default(size=20)
    label_height = 32
    sheet_paths = []
    start = 0
    for group_index, size in enumerate(group_sizes):
        columns = 1 if size == 1 else 2 if size in (2, 4) else 3
        rows = (size + columns - 1) // columns
        divisor = 1 if budget is not None else (2 if size in (1, 2, 4) else 3)
        tiles = []
        for index in range(start, start + size):
            with Image.open(checkpoint_paths[index]) as source:
                width, height = source.size
                tile = source.convert('RGB').resize(
                    (max(1, width // divisor), max(1, height // divisor)), Image.Resampling.LANCZOS
                )
            tiles.append((index, tile))
        cell_width = max(max(tile.width for _, tile in tiles), int(font.getlength(f'screenshot_{start + size - 1}')) + 16)
        image_height = max(tile.height for _, tile in tiles)
        cell_height = image_height + label_height
        sheet = Image.new('RGB', (columns * cell_width, rows * cell_height), 'white')
        draw = ImageDraw.Draw(sheet)
        for position, (index, tile) in enumerate(tiles):
            x = (position % columns) * cell_width
            y = (position // columns) * cell_height
            sheet.paste(tile, (x + (cell_width - tile.width) // 2, y))
            draw.text(
                (x + cell_width / 2, y + tile.height + 5),
                f'screenshot_{index}', fill='black', font=font, anchor='mt'
            )
        path = os.path.join(temp_folder, f'screenshot_sheet_{group_index}.png')
        sheet.save(path)
        sheet_paths.append(path)
        print(f'Screenshot sheet {group_index}: screenshots {start}–{start + size - 1}, {rows} rows × {columns} columns, scale 1/{divisor}, size {sheet.width}×{sheet.height}')
        start += size
    if budget is not None:
        records = budget['setting3_reference']['screenshots']
        areas = []
        offset = 0
        for size in group_sizes:
            areas.append(sum(r['original_size'][0] * r['original_size'][1] for r in records[offset:offset + size]))
            offset += size
        targets = allocate_targets(areas, budget['saved_pixels'], group_sizes)
        allocations = []
        for path, size, source_area, target in zip(sheet_paths, group_sizes, areas, targets):
            with Image.open(path) as image:
                raw_size = list(image.size)
            resize_to_area(path, target)
            tile_snap = snap_to_tile_boundary(path, tile_snap_margin)
            allocations.append({'screenshot_count': size, 'source_pixels': source_area,
                                'raw_sheet_size': raw_size,
                                'label_and_padding_pixels': raw_size[0] * raw_size[1] - source_area,
                                'target_pixels': target, 'assigned_saved_pixels': source_area - target,
                                'tile_snap': tile_snap})
        finish_budget(budget, sheet_paths, allocations, temp_folder, setting=2)
    return sheet_paths


def prepare_images_in_temp_folder(image_paths: List[str], checkpoint_paths: List[str], temp_folder: str, config_dict: dict, setting: int = 0, tile_snap_margin: int = 100) -> List[str]:
    if tile_snap_margin < 0:
        raise ValueError('tile_snap_margin must be nonnegative')
    if setting not in (0, 1, 2, 3):
        raise ValueError(f'Unsupported setting: {setting}')
    if not checkpoint_paths:
        raise ValueError('No screenshots found: expected checkpoints/step_<number>.png')
    screenshot_count = len(screenshot_group_sizes(len(checkpoint_paths))) if setting == 2 else len(checkpoint_paths)
    image_count = len(image_paths) + screenshot_count
    max_images = config_dict.get('max_images', 70)
    if image_count > max_images:
        raise ValueError(f'Input has {image_count} images ({len(image_paths)} assets + {screenshot_count} screenshot images), exceeding max_images={max_images}')
    os.makedirs(temp_folder, exist_ok=True)
    all_image_paths = []
    for i, img_path in enumerate(image_paths):
        filename = f"asset_{i}_{os.path.basename(img_path)}"
        temp_img_path = os.path.join(temp_folder, filename)
        shutil.copy2(img_path, temp_img_path)
        all_image_paths.append(temp_img_path)

    budget = reference_budget(checkpoint_paths, temp_folder, tile_snap_margin=tile_snap_margin) if setting in (1, 2) else None
    if setting == 2:
        all_image_paths.extend(merge_screenshots(checkpoint_paths, temp_folder, budget=budget, tile_snap_margin=tile_snap_margin))
    elif setting == 3:
        all_image_paths.extend(prepare_delta_screenshots(checkpoint_paths, temp_folder, tile_snap_margin=tile_snap_margin))

    for i, img_path in enumerate(checkpoint_paths if setting in (0, 1) else []):
        temp_img_path = os.path.join(temp_folder, f'screenshot_{i}.png')
        shutil.copy2(img_path, temp_img_path)
        all_image_paths.append(temp_img_path)

    if setting == 1:
        records = budget['setting3_reference']['screenshots']
        areas = [r['original_size'][0] * r['original_size'][1] for r in records]
        targets = [float(areas[0])] + allocate_targets(areas[1:], budget['saved_pixels'])
        screenshot_paths = all_image_paths[len(image_paths):]
        allocations = []
        for index, (path, area, target) in enumerate(zip(screenshot_paths, areas, targets)):
            if index:
                resize_to_area(path, target)
            tile_snap = snap_to_tile_boundary(path, tile_snap_margin if index else 0)
            allocations.append({'index': index, 'source_pixels': area, 'target_pixels': target,
                                'assigned_saved_pixels': area - target, 'tile_snap': tile_snap})
        finish_budget(budget, screenshot_paths, allocations, temp_folder, setting=1)

    if 'max_single_image_size' in config_dict:
        # Settings 1/2/3 use exact screenshot sizes; retain the existing size cap only for assets.
        for p in (all_image_paths if setting == 0 else all_image_paths[:len(image_paths)]):
            resize_image(p, config_dict['max_single_image_size'])

    return all_image_paths


def post_process(raw_response: str, assets_path: str) -> str:
    if not raw_response:
        return ''
    code = raw_response.strip()
    html_match = re.search(r'```html\s*\n?(.*?)\n?```', code, re.DOTALL | re.IGNORECASE)
    if html_match:
        code = html_match.group(1).strip()
    code = re.sub(r'<iframe[^>]*>.*?</iframe>', '', code, flags=re.DOTALL | re.IGNORECASE)
    code = re.sub(r'<iframe[^>]*/?>', '', code, flags=re.IGNORECASE)
    code = code.replace('```html', '')
    code = code.replace('```', '')
    code = code.replace('__PLACEHOLDER_ASSETS_BASE_DIR__', assets_path)
    return code


def mock_model_call(prompt: str, image_paths: List[str], resolution: str) -> str:
    img_tag = ''
    if image_paths:
        fname = os.path.basename(image_paths[0])
        img_tag = f'<img src="__PLACEHOLDER_ASSETS_BASE_DIR__/{fname}" width="800" height="600" alt="stitched" />'

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Generated Mock</title>
  <style>body{{font-family:Arial,Helvetica,sans-serif;padding:16px}}.preview{{max-width:1000px}}</style>
</head>
<body>
  <h1>Mock Generated Page ({resolution})</h1>
  <div class="preview">{img_tag}</div>
  <p>This is a mock HTML output produced by IWREval utils.mock_model_call. Replace with a real model call for production.</p>
</body>
</html>"""
    return html


def call_mllm(model_name: str, content_list: List[dict], timeout: int = 300, extra_params: dict | None = None, http_client=None, response_dir: str | None = None) -> str:
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL', None)
    if not api_key:
        msg = 'OPENAI_API_KEY is not set. call_mllm requires a configured API key.'
        print(msg)
        raise RuntimeError(msg)

    client = None
    if http_client is None:
        http_client = None
        try:
            proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
            if proxy:
                http_client = httpx.Client(proxies=proxy, transport=httpx.HTTPTransport(local_address='0.0.0.0'))
        except Exception as e:
            print(f"Warning: failed to configure httpx proxy client: {e}")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, http_client=http_client)

    api_params = {
        "model": model_name,
        "messages": [{"role": "user", "content": content_list}],
    }
    if extra_params:
        api_params.update(extra_params)

    try:
        print(f"Calling model {model_name}, content count: {len(content_list)}")
        response = client.chat.completions.create(**api_params)
        if response_dir is not None:
            os.makedirs(response_dir, exist_ok=True)
            response_data = response.model_dump(mode='json')
            with open(os.path.join(response_dir, 'response.json'), 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            usage = response_data.get('usage') or {}
            token_usage = {
                'requested_model': model_name,
                'returned_model': response_data.get('model'),
                'input_tokens': usage.get('prompt_tokens', usage.get('input_tokens')),
                'output_tokens': usage.get('completion_tokens', usage.get('output_tokens')),
                'total_tokens': usage.get('total_tokens'),
                'usage': response_data.get('usage'),
            }
            with open(os.path.join(response_dir, 'usage.json'), 'w', encoding='utf-8') as f:
                json.dump(token_usage, f, ensure_ascii=False, indent=2)
            print(f"Token usage: input={token_usage['input_tokens']}, output={token_usage['output_tokens']}, total={token_usage['total_tokens']} (None means not reported by API)")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Calling model API call failed: {e}")
        raise


def call_mllm_html(model_name: str, image_paths: List[str], checkpoint_paths: List[str], temp_folder: str, resolution: str, prompt: str = None, setting: int = 0, response_dir: str | None = None, tile_snap_margin: int = 100) -> str:
    config_dict = get_config(model_name)
    temp_image_paths = prepare_images_in_temp_folder(image_paths, checkpoint_paths, temp_folder, config_dict, setting=setting, tile_snap_margin=tile_snap_margin)
    # allow callers to override the default prompt template
    if prompt is None:
        prompt = PROMPT_TEMPLATE.format(resolution=resolution)

    content_list = [{"type": "text", "text": prompt}]
    delta_records = {}
    if setting == 3:
        with open(os.path.join(temp_folder, 'screenshot_manifest.json'), encoding='utf-8') as f:
            manifest = json.load(f)
        delta_records = {record['output']: record for record in manifest['screenshots']}
    for img_path in temp_image_paths:
        record = delta_records.get(os.path.basename(img_path))
        if record is not None:
            width, height = record['original_size']
            description = f"screenshot_{record['index']}: "
            if record['mode'] == 'full':
                description += f"Full webpage screenshot, original size {width}x{height}."
            else:
                x1, y1, x2, y2 = record['box']
                description += (
                    f"Crop of the current webpage state, original page size {width}x{height}; "
                    f"crop bounds (left, top, right, bottom) = ({x1}, {y1}, {x2}, {y2}), "
                    f"origin at the page's top-left, right/bottom exclusive. "
                    f"Changes were detected relative to screenshot_{record['compared_to']}; "
                    "the surrounding layout is omitted because it is visually similar within detection tolerance. "
                    "Use earlier full screenshots for the surrounding page. This is a region of the same webpage, not a separate page."
                )
            sent_width, sent_height = record['output_size']
            if record.get('tile_snap', {}).get('extra_saved_pixels', 0):
                description += (f" The supplied image has been proportionally resized to {sent_width}x{sent_height}; "
                                "all crop bounds and page dimensions above refer to the original page coordinates.")
            if record.get('theme_changes'):
                palette = '; '.join(f"{c['before']} -> {c['after']} (about {c['coverage_fraction']:.1%} of sampled pixels)"
                                    for c in record['theme_changes'])
                description += (
                    f" Broad palette changes also occurred outside the crop: {palette}. "
                    "These are coarse color correspondences, not exact element assignments; "
                    "use the previous full-page layout and current crop to infer the theme change."
                )
            content_list.append({'type': 'text', 'text': description})
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:{get_media_type(img_path)};base64,{encode_file(img_path)}"}
        })

    return call_mllm(model_name, content_list, timeout=config_dict.get('timeout', 300), extra_params=config_dict.get('extra_params', None), response_dir=response_dir)


__all__ = [
    'get_config', 'encode_file', 'get_media_type', 'resize_image', 'extract_frames',
    'find_files', 'get_resolution', 'prepare_images_in_temp_folder', 'post_process', 'mock_model_call', 'call_mllm', 'call_mllm_html'
]
