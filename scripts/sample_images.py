#!/usr/bin/env python3
"""Sample images into the directory layout expected by demo.py."""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def natural_sort_key(path: Path) -> list[object]:
    """Sort frame_2 before frame_10 while remaining case-insensitive."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.as_posix())
    ]


def find_images(source_dir: Path, recursive: bool) -> list[Path]:
    candidates: Iterable[Path]
    candidates = source_dir.rglob("*") if recursive else source_dir.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )


def select_images(
    images: list[Path], count: int, strategy: str, seed: int
) -> list[Path]:
    sample_count = min(count, len(images))

    if strategy == "first":
        indices = list(range(sample_count))
    elif strategy == "random":
        indices = sorted(random.Random(seed).sample(range(len(images)), sample_count))
    elif sample_count == 1:
        indices = [len(images) // 2]
    else:
        # Include both ends and distribute the remaining samples over the full
        # sequence. This is usually more useful for multi-view reconstruction
        # than taking adjacent video frames.
        indices = [
            round(i * (len(images) - 1) / (sample_count - 1))
            for i in range(sample_count)
        ]

    return [images[index] for index in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample images and create TARGET_DIR/images, the input layout "
            "expected by demo.py."
        )
    )
    parser.add_argument("source_dir", type=Path, help="directory containing source images")
    parser.add_argument("target_dir", type=Path, help="new demo scene directory")
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=10,
        help="number of images to sample (default: 10)",
    )
    parser.add_argument(
        "--strategy",
        choices=("uniform", "first", "random"),
        default="uniform",
        help="sampling strategy (default: uniform)",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="link or copy sampled images (default: symlink)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed used by --strategy random (default: 0)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="also search subdirectories of SOURCE_DIR",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    images_dir = target_dir / "images"

    if args.count <= 0:
        print("error: --count must be greater than zero", file=sys.stderr)
        return 2
    if not source_dir.is_dir():
        print(f"error: source directory does not exist: {source_dir}", file=sys.stderr)
        return 2

    images = find_images(source_dir, args.recursive)
    if not images:
        print(f"error: no supported images found in: {source_dir}", file=sys.stderr)
        return 1

    if images_dir.exists() and any(images_dir.iterdir()):
        print(
            f"error: target image directory is not empty: {images_dir}\n"
            "Use a new TARGET_DIR to avoid mixing different samples.",
            file=sys.stderr,
        )
        return 1

    selected = select_images(images, args.count, args.strategy, args.seed)
    selected_names = [path.name for path in selected]
    duplicate_names = sorted(
        name for name, occurrences in Counter(selected_names).items() if occurrences > 1
    )
    if duplicate_names:
        print(
            "error: duplicate filenames found while sampling recursively: "
            + ", ".join(duplicate_names),
            file=sys.stderr,
        )
        return 1

    images_dir.mkdir(parents=True, exist_ok=True)
    for source in selected:
        destination = images_dir / source.name
        if args.mode == "symlink":
            destination.symlink_to(source)
        else:
            shutil.copy2(source, destination)

    manifest_path = target_dir / "sample_manifest.txt"
    manifest_path.write_text(
        "\n".join(str(path) for path in selected) + "\n", encoding="utf-8"
    )

    if len(images) < args.count:
        print(
            f"warning: requested {args.count} images, but only {len(images)} were found; "
            f"selected all {len(selected)} images."
        )

    print(f"Found:    {len(images)} images")
    print(f"Selected: {len(selected)} images ({args.strategy})")
    print(f"Mode:     {args.mode}")
    print(f"Output:   {images_dir}")
    print(f"Manifest: {manifest_path}")
    for index, path in enumerate(selected, start=1):
        print(f"  {index:03d}: {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
