#!/usr/bin/env python3
"""Build a local gallery of setting 3 results, without generation or API calls."""

import argparse
import html
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont

from utils.screenshot_delta import DeltaOptions, prepare_delta_screenshots
from utils.screenshot_delta_adaptive import AdaptiveDeltaOptions, prepare_delta_screenshots as prepare_adaptive


ROOT = Path(__file__).resolve().parent
STYLE = '''body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#172033}
a{color:#145db5}article{background:white;padding:20px;margin:20px 0;border-radius:12px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.pair img{max-width:100%;max-height:360px;object-fit:contain}
table{border-collapse:collapse;background:white}td,th{padding:10px 18px;border:1px solid #ccd2dd;text-align:left}
code{overflow-wrap:anywhere}small{color:#536076}@media(max-width:700px){.pair{grid-template-columns:1fr}}'''


def write_page(path, title, body):
    path.write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                    f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                    f'<title>{html.escape(title)}</title><style>{STYLE}</style>'
                    f'<body><h1>{html.escape(title)}</h1>{body}</body></html>', encoding='utf-8')


def link(path, page_dir):
    return html.escape(quote(os.path.relpath(path, page_dir).replace(os.sep, '/')), quote=True)


def render_task(task_dir, output_dir, options, algorithm='adaptive', tile_snap_margin=100):
    paths = sorted((int(match.group(1)), str(path))
                   for path in (task_dir / 'checkpoints').glob('step_*.png')
                   if (match := re.fullmatch(r'step_(\d+)\.png', path.name)))
    if not paths:
        return None
    source_paths = [path for _, path in paths]
    images_dir = output_dir / 'temp_images'
    prepare = prepare_adaptive if algorithm != 'baseline' else prepare_delta_screenshots
    if algorithm == 'baseline':
        prepare(source_paths, images_dir, options)
    else:
        prepare(source_paths, images_dir, options, tile_snap_margin=tile_snap_margin)
    manifest = json.loads((images_dir / 'screenshot_manifest.json').read_text(encoding='utf-8'))
    records = manifest['screenshots']
    overview = Image.new('RGB', (1280, 400 * len(records)), 'white')
    font = ImageFont.load_default(size=18)
    body = [f'<p>算法版本：<strong>{html.escape(manifest["algorithm"])}</strong>。新版算法已用于生成 setting 3。</p>',
            '<p><a href="../index.html">返回任务总览</a> · '
            '<a href="preview.png">查看全部步骤拼图</a> · '
            '<a href="temp_images/screenshot_manifest.json">查看算法判断 JSON</a></p>',
            '<p>左侧为原图预览和选区（绿色：裁剪；蓝色：保留整图），右侧为此算法的输出预览。'
            '点击图片查看原尺寸；预览自动缩放。像素减少比例不等于 token 减少比例。</p>']
    total_pixels = retained_pixels = 0
    for record in records:
        index = record['index']
        with Image.open(record['source']) as original:
            thumb = original.convert('RGB')
        width, height = thumb.size
        thumb.thumbnail((620, 330))
        sx, sy = thumb.width / width, thumb.height / height
        x1, y1, x2, y2 = record['box']
        draw = ImageDraw.Draw(thumb)
        draw.rectangle((x1 * sx, y1 * sy, min(thumb.width - 1, x2 * sx), min(thumb.height - 1, y2 * sy)),
                       outline='#138a36' if record['mode'] == 'crop' else '#145db5', width=3)
        marked = output_dir / f'original_marked_{index}.png'
        thumb.save(marked)
        result_path = images_dir / record['output']
        with Image.open(result_path) as result:
            result_thumb = result.convert('RGB')
        result_thumb.thumbnail((620, 330))
        overview.paste(thumb, (10, 400 * index + 40))
        overview.paste(result_thumb, (650, 400 * index + 40))
        draw = ImageDraw.Draw(overview)
        label = f"screenshot_{index}: {record['mode']} | {record['original_size']} -> {record['output_size']}"
        draw.text((10, 400 * index + 10), label, fill='black', font=font)
        draw.text((10, 400 * index + 375), f"box={record['box']} | {record['reason']}", fill='black', font=font)
        total_pixels += width * height
        retained_pixels += record['output_size'][0] * record['output_size'][1]
        metrics = html.escape(json.dumps(record['metrics'], ensure_ascii=False))
        palette = ''
        for change in record.get('theme_changes', []):
            before, after = change['before'], change['after']
            palette += (f'<span style="display:inline-block;background:{before};width:24px;height:18px;border:1px solid #aaa"></span>'
                        f' {before} → '
                        f'<span style="display:inline-block;background:{after};width:24px;height:18px;border:1px solid #aaa"></span>'
                        f' {after}（约 {change["coverage_fraction"]:.1%}） ')
        body.append(f'<article><h2>screenshot_{index} · {record["mode"]}</h2>'
                    f'<p>{record["original_size"]} → {record["output_size"]} · '
                    f'保留像素 {record["retained_pixel_fraction"]:.1%} · 场景 {record["scene_id"]}</p>'
                    f'<div class="pair"><div><p>原图与选区</p><a href="{link(Path(record["source"]), output_dir)}">'
                    f'<img loading="lazy" src="{link(marked, output_dir)}"></a></div>'
                    f'<div><p>算法输出预览</p><a href="{link(result_path, output_dir)}">'
                    f'<img loading="lazy" src="{link(result_path, output_dir)}"></a></div></div>'
                    f'<p>坐标：<code>{record["box"]}</code> · 原因：{record["reason"]}</p>'
                    f'<small>相邻原图比较指标：{metrics}</small>'
                    f'<p>{"主题配色变化（裁剪之外也可能发生）：" + palette if palette else ""}</p></article>')
    overview.save(output_dir / 'preview.png')
    write_page(output_dir / 'index.html', f'Setting 3 · 任务 {task_dir.name}', ''.join(body))
    return {'task': task_dir.name, 'screenshots': len(records),
            'full': sum(r['mode'] == 'full' for r in records),
            'crop': sum(r['mode'] == 'crop' for r in records),
            'pixel_reduction': 1 - retained_pixels / total_pixels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/data_backup')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/setting3_previews')
    parser.add_argument('--task', nargs='+', help='Optional task IDs, e.g. --task 1 2')
    parser.add_argument('--pixel-threshold', type=int, default=18)
    parser.add_argument('--padding', type=int, default=24)
    parser.add_argument('--tile-snap-margin', type=int, default=100)
    parser.add_argument('--algorithm', choices=('adaptive', 'experimental', 'baseline'), default='adaptive',
                        help='Current adaptive algorithm (default), legacy baseline, or experimental as an alias for adaptive')
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error(f'Data root not found: {args.data_root}')
    if args.tile_snap_margin < 0:
        parser.error('tile-snap-margin must be nonnegative')
    if not 0 <= args.pixel_threshold <= 255 or args.padding < 0:
        parser.error('pixel-threshold must be 0–255 and padding must be nonnegative')
    if args.output.resolve() == args.data_root.resolve() or args.output.resolve() in args.data_root.resolve().parents:
        parser.error('Output must not equal or contain the data root')
    tasks = sorted((p for p in args.data_root.iterdir() if p.is_dir() and (p / 'checkpoints').is_dir()),
                   key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name))
    if args.task:
        selected = set(args.task)
        missing = selected - {task.name for task in tasks}
        if missing:
            parser.error(f'Tasks with checkpoints not found: {sorted(missing)}')
        tasks = [task for task in tasks if task.name in selected]
    if not tasks:
        parser.error('No task/checkpoints directories found')
    args.output.mkdir(parents=True, exist_ok=True)
    option_type = AdaptiveDeltaOptions if args.algorithm != 'baseline' else DeltaOptions
    options = option_type(pixel_threshold=args.pixel_threshold, padding=args.padding)
    results, errors = [], []
    for task in tasks:
        try:
            result = render_task(task, args.output / task.name, options, args.algorithm, args.tile_snap_margin)
            if result is not None:
                results.append(result)
            else:
                errors.append({'task': task.name, 'error': 'No step_<number>.png images'})
        except Exception as error:
            errors.append({'task': task.name, 'error': str(error)})
            print(f'Task {task.name} failed: {error}')
    rows = [f'<p>预览算法：{args.algorithm}。adaptive 对应当前 setting 3；baseline 为旧版。</p>',
            '<p>仅做本地图片处理，不调用模型 API。点击任务编号查看逐步对比。'
            '本次总览仅列出本次选中的任务。</p><table><tr><th>任务</th><th>截图数</th>'
            '<th>整图</th><th>局部图</th><th>截图像素减少</th></tr>']
    for result in results:
        name = html.escape(result['task'])
        url = link(args.output / result['task'] / 'index.html', args.output)
        rows.append(f'<tr><td><a href="{url}">{name}</a></td><td>{result["screenshots"]}</td>'
                    f'<td>{result["full"]}</td><td>{result["crop"]}</td><td>{result["pixel_reduction"]:.1%}</td></tr>')
    rows.append('</table>')
    for error in errors:
        rows.append(f'<p>任务 {html.escape(error["task"])} 失败：{html.escape(error["error"])}</p>')
    write_page(args.output / 'index.html', 'Setting 3 裁剪预览', ''.join(rows))
    (args.output / 'summary.json').write_text(json.dumps({'algorithm': args.algorithm, 'tile_snap_margin': args.tile_snap_margin, 'options': vars(options), 'tasks': results, 'errors': errors},
                                                       ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Finished: {len(results)} tasks, {len(errors)} errors. Open {args.output / "index.html"}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
