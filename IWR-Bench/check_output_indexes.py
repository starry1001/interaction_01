#!/usr/bin/env python3
"""检查 outputs/1 到 outputs/50 中各 setting 的 index.html 是否存在。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outputs", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="outputs 目录，默认使用脚本所在目录下的 outputs",
    )
    args = parser.parse_args()
    root = args.outputs.resolve()
    if not root.is_dir():
        parser.error(f"目录不存在：{root}")

    missing = []
    for sample_id in range(1, 51):
        for setting in range(4):
            path = root / str(sample_id) / f"gpt-6-luna_setting{setting}" / "index.html"
            if not path.is_file():
                missing.append(path)

    print(f"检查目录：{root}")
    print(f"共检查 50 × 4 = 200 个 index.html，存在 {200 - len(missing)} 个，缺少 {len(missing)} 个。")
    for path in missing:
        reason = "（setting 文件夹不存在）" if not path.parent.is_dir() else ""
        print(f"缺少：{path.relative_to(root)}{reason}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
