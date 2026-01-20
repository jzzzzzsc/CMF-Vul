#!/usr/bin/env python3
"""
Render all Graphviz .dot files in a directory to PNG images.

Example:
  python render_dot_to_png.py --dot-dir /path/to/dots --out-dir /path/to/pngs
"""

import argparse
import os
from pathlib import Path

from graphviz import Source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Graphviz .dot files to PNG images."
    )
    parser.add_argument(
        "--dot-dir",
        required=True,
        type=Path,
        help="Directory containing .dot files.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to save rendered PNG files.",
    )
    parser.add_argument(
        "--sort",
        choices=["asc", "desc"],
        default="asc",
        help="Sort .dot files by size (asc or desc). Default: asc.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files (default: skip if exists).",
    )
    return parser.parse_args()


def list_dot_files(dot_dir: Path, ascending: bool) -> list[tuple[Path, int]]:
    dot_files = [p for p in dot_dir.iterdir() if p.is_file() and p.suffix == ".dot"]
    dot_files_with_size = [(p, p.stat().st_size) for p in dot_files]
    dot_files_with_size.sort(key=lambda x: x[1], reverse=not ascending)
    return dot_files_with_size


def render_dot_to_png(dot_path: Path, out_dir: Path, overwrite: bool) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Graphviz.Source.render expects an output "filename" without extension.
    output_stem = out_dir / dot_path.stem
    png_path = output_stem.with_suffix(".png")

    if png_path.exists() and not overwrite:
        print(f"Skip (already exists): {png_path}")
        return True

    try:
        src = Source.from_file(str(dot_path))
        src.render(filename=str(output_stem), format="png", cleanup=True)
    except Exception as e:
        print(f"Failed: {dot_path} ({e})")
        return False

    if png_path.exists():
        print(f"OK: {png_path}")
        return True

    print(f"Failed (PNG not found): {png_path}")
    return False


def main() -> None:
    args = parse_args()

    dot_dir: Path = args.dot_dir
    out_dir: Path = args.out_dir

    if not dot_dir.exists() or not dot_dir.is_dir():
        raise SystemExit(f"--dot-dir must be an existing directory: {dot_dir}")

    ascending = (args.sort == "asc")
    dot_files_with_size = list_dot_files(dot_dir, ascending=ascending)

    if not dot_files_with_size:
        print(f"No .dot files found in: {dot_dir}")
        return

    total = len(dot_files_with_size)
    success = 0

    for i, (dot_path, size) in enumerate(dot_files_with_size, start=1):
        print(f"[{i}/{total}] Rendering: {dot_path.name} (size={size} bytes)")
        if render_dot_to_png(dot_path, out_dir, overwrite=args.overwrite):
            success += 1

    print(f"Done. Success: {success}/{total}. Output: {out_dir}")


if __name__ == "__main__":
    main()
