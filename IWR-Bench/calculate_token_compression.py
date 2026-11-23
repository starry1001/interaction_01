#!/usr/bin/env python3
"""按样本配对，统计 setting1/2/3 相对 setting0 的 input_tokens 压缩率。"""

import argparse
import json
from pathlib import Path
from statistics import mean


def read_tokens(path):
    """只读取顶层 input_tokens；缺失或无效的数据不参与统计。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["input_tokens"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"{path}: {exc}"
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None, f"{path}: input_tokens 必须是非负整数"
    return value, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outputs", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="数据目录，默认使用脚本所在目录下的 outputs",
    )
    parser.add_argument("--model", default="gpt-6-luna", help="目录中的模型名称")
    args = parser.parse_args()
    root = args.outputs.resolve()
    if not root.is_dir():
        parser.error(f"目录不存在：{root}")
    samples = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not samples:
        parser.error(f"没有找到数字编号的样本目录：{root}")

    tokens = {}
    errors = []
    for sample in samples:
        for setting in range(4):
            path = sample / f"{args.model}_setting{setting}" / "usage.json"
            value, error = read_tokens(path)
            tokens[sample.name, setting] = value
            if error:
                errors.append(error)

    print(f"数据目录：{root}")
    print(f"模型：{args.model}；样本目录数：{len(samples)}")
    print("平均压缩率 = mean(1 - settingN / setting0)，每个样本权重相同")
    print("总量压缩率 = 1 - sum(settingN) / sum(setting0)，使用同一批配对样本")
    print("正值表示节省，负值表示增加。setting0 为 0 的样本跳过。\n")
    print("setting   配对数   setting0平均tokens   当前平均tokens   平均压缩率   总量压缩率")
    has_pairs = False
    for setting in range(1, 4):
        pairs = []
        skipped = []
        for sample in samples:
            baseline = tokens[sample.name, 0]
            current = tokens[sample.name, setting]
            if baseline is None or baseline == 0 or current is None:
                skipped.append(sample.name)
            else:
                pairs.append((baseline, current))
        if pairs:
            has_pairs = True
            base_total = sum(b for b, _ in pairs)
            current_total = sum(c for _, c in pairs)
            average_reduction = mean(1 - c / b for b, c in pairs)
            total_reduction = 1 - current_total / base_total
            print(
                f"setting{setting}  {len(pairs):>6}   {base_total / len(pairs):>17,.2f}"
                f"   {current_total / len(pairs):>13,.2f}"
                f"   {average_reduction:>10.2%}   {total_reduction:>10.2%}"
            )
        else:
            print(f"setting{setting}：无有效配对数据")
        if skipped:
            print(f"  跳过样本：{', '.join(skipped)}")
    if errors:
        print(f"\n缺失或无效文件（{len(errors)} 个）：")
        for error in errors:
            print(f"  {error}")
    return 0 if has_pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
