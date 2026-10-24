#!/usr/bin/env python3
"""将 TFLite 模型转换为 TFLite Micro 固件所需的 C 字节数组。"""

from __future__ import annotations

import argparse
from pathlib import Path


def convert_model(input_path: Path, output_path: Path) -> None:
    """把二进制模型写成带固定符号名的 C++ 源文件。"""
    data = input_path.read_bytes()
    rows = []
    for offset in range(0, len(data), 12):
        chunk = data[offset : offset + 12]
        values = ", ".join(f"0x{value:02x}" for value in chunk)
        rows.append(f"  {values},")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "// 由 tools/tflite_to_c.py 自动生成。\n"
        "\n"
        "extern const unsigned char g_cifar_model[] = {\n"
        + "\n".join(rows)
        + "\n};\n"
        f"extern const unsigned int g_cifar_model_len = {len(data)};\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert_model(args.input, args.output)


if __name__ == "__main__":
    main()
