import os
import base64
import shutil
import json
import re
import math
from typing import List, Tuple
from config import API_TIMEOUT, PROMPT_TEMPLATE
from openai import OpenAI
import httpx
from PIL import Image, ImageDraw, ImageFont
from setting3 import prepare_delta_screenshots
from .pixel_budget import estimate_image_tokens, resize_to_token_budget, write_budget, original_detail_for_model, validate_original_size


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


def merge_screenshots(checkpoint_paths: List[str], temp_folder: str) -> List[str]:
    """Keep existing groups; stack landscape images down and portrait images across."""
    os.makedirs(temp_folder, exist_ok=True)
    font = ImageFont.load_default(size=20)
    sheet_paths, allocations = [], []
    start = 0
    for group_index, count in enumerate(screenshot_group_sizes(len(checkpoint_paths))):
        entries = []
        for index in range(start, start + count):
            with Image.open(checkpoint_paths[index]) as image:
                width, height = image.size
            label = f'screenshot_{index}'
            entries.append({'index': index, 'size': [width, height], 'label': label,
                            'cell_width': max(width, math.ceil(font.getlength(label)) + 16),
                            'cell_height': height + 32})
        # Mixed orientations: majority wins; ties follow the first frame. No rotation.
        wide = sum(e['size'][0] >= e['size'][1] for e in entries)
        vertical = wide * 2 > count or (wide * 2 == count and entries[0]['size'][0] >= entries[0]['size'][1])
        if vertical:
            canvas_size = (max(e['cell_width'] for e in entries), sum(e['cell_height'] for e in entries))
        else:
            canvas_size = (sum(e['cell_width'] for e in entries), max(e['cell_height'] for e in entries))
        sheet = Image.new('RGB', canvas_size, 'white')
        draw = ImageDraw.Draw(sheet)
        x = y = 0
        regions = []
        for entry in entries:
            index = entry['index']
            width, height = entry['size']
            cell_width = canvas_size[0] if vertical else entry['cell_width']
            image_x = x + (cell_width - width) // 2
            with Image.open(checkpoint_paths[index]) as source:
                sheet.paste(source.convert('RGB'), (image_x, y))
            draw.text((x + cell_width / 2, y + height + 5), entry['label'], fill='black', font=font, anchor='mt')
            regions.append({'index': index, 'source_size': entry['size'],
                            'source_sheet_box': [image_x, y, image_x + width, y + height]})
            if vertical:
                y += entry['cell_height']
            else:
                x += entry['cell_width']
        path = os.path.join(temp_folder, f'screenshot_sheet_{group_index}.png')
        sheet.save(path, compress_level=1)
        sheet.close()
        reference_tokens = sum(estimate_image_tokens(*entry['size']) for entry in entries)
        record = resize_to_token_budget(path, reference_tokens=reference_tokens)
        record.update(screenshot_count=count, source_indices=list(range(start, start + count)),
                      layout='vertical' if vertical else 'horizontal', source_regions=regions,
                      original_screenshot_pixels=sum(e['size'][0] * e['size'][1] for e in entries))
        sx, sy = record['output_size'][0] / canvas_size[0], record['output_size'][1] / canvas_size[1]
        for region in regions:
            a, b, c, d = region['source_sheet_box']
            region['output_sheet_box'] = [math.floor(a*sx), math.floor(b*sy), math.ceil(c*sx), math.ceil(d*sy)]
        allocations.append(record)
        sheet_paths.append(path)
        print(f'Screenshot sheet {group_index}: screenshots {start}–{start + count - 1}, {record["layout"]}, {canvas_size} -> {record["output_size"]}')
        start += count
    write_budget(allocations, temp_folder, setting=2)
    return sheet_paths


def prepare_images_in_temp_folder(image_paths: List[str], checkpoint_paths: List[str], temp_folder: str, setting: int = 0) -> List[str]:
    if setting not in (0, 1, 2, 3):
        raise ValueError(f'Unsupported setting: {setting}')
    if not checkpoint_paths:
        raise ValueError('No screenshots found: expected checkpoints/step_<number>.png')
    os.makedirs(temp_folder, exist_ok=True)
    all_image_paths = []
    for index, img_path in enumerate(image_paths):
        temp_img_path = os.path.join(temp_folder, f'asset_{index}_{os.path.basename(img_path)}')
        shutil.copy2(img_path, temp_img_path)
        all_image_paths.append(temp_img_path)

    if setting == 2:
        all_image_paths.extend(merge_screenshots(checkpoint_paths, temp_folder))
    elif setting == 3:
        all_image_paths.extend(prepare_delta_screenshots(checkpoint_paths, temp_folder))
    else:
        allocations = []
        for index, source in enumerate(checkpoint_paths):
            destination = os.path.join(temp_folder, f'screenshot_{index}.png')
            shutil.copy2(source, destination)
            all_image_paths.append(destination)
            if setting == 1:
                record = resize_to_token_budget(destination)
                record['index'] = index
                allocations.append(record)
        if setting == 1:
            write_budget(allocations, temp_folder, setting=1)
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


def call_mllm(model_name: str, content_list: List[dict], timeout: int = API_TIMEOUT, extra_params: dict | None = None, http_client=None, response_dir: str | None = None) -> str:
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


def call_mllm_html(model_name: str, image_paths: List[str], checkpoint_paths: List[str], temp_folder: str, resolution: str, prompt: str = None, setting: int = 0, response_dir: str | None = None) -> str:
    temp_image_paths = prepare_images_in_temp_folder(image_paths, checkpoint_paths, temp_folder, setting=setting)
    # allow callers to override the default prompt template
    if prompt is None:
        prompt = PROMPT_TEMPLATE.format(resolution=resolution)

    content_list = [{"type": "text", "text": prompt}]
    for img_path in temp_image_paths:
        image_url = {"url": f"data:{get_media_type(img_path)};base64,{encode_file(img_path)}"}
        if original_detail_for_model(model_name):
            with Image.open(img_path) as image:
                validate_original_size(*image.size)
            image_url['detail'] = 'original'
        content_list.append({"type": "image_url", "image_url": image_url})

    return call_mllm(model_name, content_list, response_dir=response_dir)


__all__ = [
    'encode_file', 'get_media_type',
    'find_files', 'get_resolution', 'prepare_images_in_temp_folder', 'post_process', 'mock_model_call', 'call_mllm', 'call_mllm_html'
]
