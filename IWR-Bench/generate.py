#!/usr/bin/env python3
import os
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from utils import core as utils

load_dotenv()


def process_task(task_item, data_dir: str, result_dir: str, model_name: str, coverage: bool = False, setting: int = 0, tile_snap_margin: int = 100):
	if setting not in (0, 1, 2, 3):
		raise ValueError(f'Unsupported setting: {setting}')
	task_id = str(task_item.get('id'))


	task_path = os.path.join(data_dir, task_id)
	print(f"Processing task: {task_path}")
	stitched_images, checkpoint_images = utils.find_files(task_path)

	if not checkpoint_images:
		return task_id, "No screenshots found in checkpoints/step_<number>.png"

	if isinstance(task_item, dict):
		resolution = task_item.get('recording_details', {}).get('resolution') or utils.get_resolution(task_path)
	else:
		resolution = utils.get_resolution(task_path)

	model_dir = f"{model_name.lower().replace('/', '__')}_setting{setting}"
	output_dir = os.path.join(result_dir, task_id, model_dir)
	os.makedirs(output_dir, exist_ok=True)
	output_file = os.path.join(output_dir, 'index.html')

	if not coverage and os.path.exists(output_file):
		return task_id, 'Already exists'

	temp_folder = os.path.join(output_dir, 'temp_images')
	if os.path.exists(temp_folder):
		shutil.rmtree(temp_folder)

	print(f"Calling model for task {task_id} (model={model_name}, setting={setting})")
	screenshot_count = len(utils.screenshot_group_sizes(len(checkpoint_images))) if setting == 2 else len(checkpoint_images)
	print(f"Input: {len(stitched_images)} stitched assets followed by {screenshot_count} screenshot images ({len(checkpoint_images)} source screenshots)")
	if model_name == 'mock':
		config = utils.get_config(model_name)
		temp_images = utils.prepare_images_in_temp_folder(stitched_images, checkpoint_images, temp_folder, config, setting=setting, tile_snap_margin=tile_snap_margin)
		raw = utils.mock_model_call("", temp_images, resolution)
	else:
		raw = utils.call_mllm_html(model_name, stitched_images, checkpoint_images, temp_folder, resolution, setting=setting, response_dir=output_dir, tile_snap_margin=tile_snap_margin)

	# Preserve the exact model text before extracting HTML or replacing asset paths.
	Path(output_dir, 'raw_response.txt').write_text(raw or '', encoding='utf-8')
	if setting == 3:
		shutil.copy2(Path(temp_folder) / 'screenshot_manifest.json', Path(output_dir) / 'screenshot_manifest.json')
	if setting in (1, 2):
		shutil.copy2(Path(temp_folder) / 'compression_budget.json', Path(output_dir) / 'compression_budget.json')

	assets_dir = os.path.join(data_dir, task_id, 'assets')
	assets_path = os.path.relpath(assets_dir, output_dir).replace('\\', '/')
	html = utils.post_process(raw, assets_path)
	if not html:
		raise ValueError(f'Empty generated HTML for task {task_id}; raw response saved in {output_dir}')

	# Keep an independent snapshot for each generation, including repeated --coverage runs.
	backup_dir = Path(output_dir) / 'backups' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
	backup_dir.mkdir(parents=True, exist_ok=False)
	(backup_dir / 'index.html').write_text(html, encoding='utf-8')
	artifact_names = ['raw_response.txt'] + (['response.json', 'usage.json'] if model_name != 'mock' else [])
	if setting == 3:
		artifact_names.append('screenshot_manifest.json')
	if setting in (1, 2):
		artifact_names.append('compression_budget.json')
	for name in artifact_names:
		shutil.copy2(Path(output_dir) / name, backup_dir / name)

	with open(output_file, 'w', encoding='utf-8') as f:
		f.write(html)

	return task_id, 'Done'



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', required=True, help='Path to dataset directory containing task folders or an example.jsonl file')
	parser.add_argument('--result', required=True, help='Path to write generated outputs')
	parser.add_argument('--model', default='mock', help='Model name (mock by default)')
	parser.add_argument('--setting', type=int, choices=(0, 1, 2, 3), default=0,
		help='0: direct; 1: match setting3 pixel budget, keep first screenshot; 2: merge at original size then match setting3 pixel budget; 3: adaptive scene/change crops')
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--task', '--ids', nargs='+', help='Only run these task IDs, e.g. --task 3 or --ids 1 3')
	parser.add_argument('--tile-snap-margin', type=int, default=100,
		help='Settings 1/2/3: snap down near 512px boundaries within this many pixels; 0 disables (default: 100)')
	parser.add_argument('--coverage', action='store_true', help='If set, regenerate and replace existing index.html after successful generation')
	args = parser.parse_args()
	if args.tile_snap_margin < 0:
		parser.error('--tile-snap-margin must be nonnegative')

	jsonl_path = args.data
	result_path = args.result

	base = os.path.splitext(os.path.basename(jsonl_path))[0]
	data_root = os.path.join(os.path.dirname(jsonl_path), base)
	if not os.path.isdir(data_root):
		print(f"Tasks directory not found next to {jsonl_path}: expected {data_root}")
		return

	tasks = utils.load_examples(jsonl_path)
	if args.task:
		selected = set(args.task)
		missing = selected - {str(task.get('id')) for task in tasks}
		if missing:
			parser.error(f'Task IDs not found in dataset: {sorted(missing)}')
		tasks = [task for task in tasks if str(task.get('id')) in selected]
	if not tasks:
		print('No tasks found')
		return

	print(f"Processing {len(tasks)} tasks with {args.workers} workers using model {args.model}, setting {args.setting}")

	with ThreadPoolExecutor(max_workers=args.workers) as ex:
		ordered = sorted(tasks, key=lambda x: (x.get('id') if isinstance(x, dict) else x))
		futures = {ex.submit(process_task, t, data_root, result_path, args.model, args.coverage, args.setting, args.tile_snap_margin): t for t in ordered}
		completed = 0
		for future in as_completed(futures):
			task, status = future.result()
			completed += 1
			print(f"[{completed}/{len(tasks)}] {task}: {status}")


if __name__ == '__main__':
	main()
