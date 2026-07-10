#!/usr/bin/env python3
"""Export every IGGT instance cluster as an individual colored PLY file."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split IGGT predictions into one colored PLY point cloud per "
            "DBSCAN/HDBSCAN instance label."
        )
    )
    parser.add_argument("predictions", type=Path, help="demo.py predictions.npz file")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="output directory (default: a sibling directory named instances)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help=(
            "directory containing mask_*.npy labels; only needed for old "
            "predictions.npz files without instance_labels"
        ),
    )
    parser.add_argument(
        "--color-mode",
        choices=("rgb", "mask"),
        default="rgb",
        help="use source-image or clustering colors (default: rgb)",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=0.3,
        help="discard points below this confidence percentile (default: 0.3)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=100,
        help="skip instances with fewer valid points (default: 100)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="randomly limit each instance to this many points; 0 keeps all",
    )
    parser.add_argument("--seed", type=int, default=0, help="downsampling seed")
    parser.add_argument(
        "--raw-coordinates",
        action="store_true",
        help="do not apply the same first-camera alignment used by GLB/PLY export",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing instance_*.ply files in OUTPUT_DIR",
    )
    return parser.parse_args()


def remove_singleton_batch(array, target_ndim: int):
    while array.ndim > target_ndim and array.shape[0] == 1:
        array = array[0]
    return array


def load_labels(data, predictions_path: Path, labels_dir: Path | None, np):
    if "instance_labels" in data.files:
        return remove_singleton_batch(np.asarray(data["instance_labels"]), 3)

    labels_dir = labels_dir or predictions_path.parent / "dbscan_masks"
    label_paths = sorted(labels_dir.glob("mask_*.npy"))
    if not label_paths:
        raise ValueError(
            "No instance_labels key or dbscan_masks/mask_*.npy files were found."
        )
    return np.stack([np.load(path) for path in label_paths])


def image_colors(images, expected_shape: tuple[int, ...], np):
    images = remove_singleton_batch(np.asarray(images), 4)
    if images.ndim != 4:
        raise ValueError(f"Expected a 4D image array, got shape {images.shape}.")
    if images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    if images.shape[:3] != expected_shape or images.shape[-1] < 3:
        raise ValueError(
            f"Image/label shape mismatch: images={images.shape}, labels={expected_shape}."
        )
    return images[..., :3]


def scene_alignment_matrix(extrinsic, np):
    extrinsic = remove_singleton_batch(np.asarray(extrinsic), 3)
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsic shape (S, 3, 4), got {extrinsic.shape}.")

    first_extrinsic = np.eye(4, dtype=np.float64)
    first_extrinsic[:3, :4] = extrinsic[0]
    opengl_conversion = np.diag([1.0, -1.0, -1.0, 1.0])
    align_y_180 = np.diag([-1.0, 1.0, -1.0, 1.0])
    return np.linalg.inv(first_extrinsic) @ opengl_conversion @ align_y_180


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else predictions_path.parent / "instances"
    )
    labels_dir = args.labels_dir.expanduser().resolve() if args.labels_dir else None

    if not predictions_path.is_file():
        print(f"error: predictions file does not exist: {predictions_path}", file=sys.stderr)
        return 2
    if predictions_path.suffix.lower() != ".npz":
        print("error: predictions must be a .npz file", file=sys.stderr)
        return 2
    if not 0.0 <= args.confidence_percentile <= 100.0:
        print("error: --confidence-percentile must be between 0 and 100", file=sys.stderr)
        return 2
    if args.min_points <= 0 or args.max_points < 0:
        print("error: point limits must be non-negative and --min-points > 0", file=sys.stderr)
        return 2

    try:
        import numpy as np
        import trimesh
        from iggt.utils.pointcloud_io import export_colored_point_cloud
    except ImportError as exc:
        print(
            f"error: missing export dependency: {exc}\n"
            "Run this script in the IGGT environment.",
            file=sys.stderr,
        )
        return 1

    try:
        with np.load(predictions_path, allow_pickle=True) as data:
            labels = load_labels(data, predictions_path, labels_dir, np).astype(np.int64)
            point_key = "world_points" if "world_points" in data.files else "world_points_from_depth"
            points = remove_singleton_batch(np.asarray(data[point_key]), 4)
            confidence_key = (
                "world_points_conf" if "world_points_conf" in data.files else "depth_conf"
            )
            confidence = remove_singleton_batch(np.asarray(data[confidence_key]), 3)
            if confidence.ndim == 4 and confidence.shape[-1] == 1:
                confidence = confidence[..., 0]

            color_key = "images" if args.color_mode == "rgb" else "features"
            if color_key not in data.files:
                raise ValueError(f"Missing {color_key!r} in {predictions_path}.")
            colors = image_colors(data[color_key], tuple(labels.shape), np)
            alignment = (
                None
                if args.raw_coordinates
                else scene_alignment_matrix(data["extrinsic"], np)
            )
    except (KeyError, ValueError) as exc:
        print(f"error: invalid prediction data: {exc}", file=sys.stderr)
        return 1

    if points.ndim != 4 or points.shape[-1] != 3:
        print(f"error: expected point shape (S, H, W, 3), got {points.shape}", file=sys.stderr)
        return 1
    if points.shape[:3] != labels.shape or confidence.shape != labels.shape:
        print(
            "error: point/label/confidence shapes do not match: "
            f"points={points.shape}, labels={labels.shape}, confidence={confidence.shape}",
            file=sys.stderr,
        )
        return 1

    existing_files = sorted(output_dir.glob("instance_*.ply")) if output_dir.exists() else []
    if existing_files and not args.overwrite:
        print(
            f"error: {output_dir} already contains instance PLY files. "
            "Use a new directory or pass --overwrite.",
            file=sys.stderr,
        )
        return 1
    if args.overwrite:
        for path in existing_files:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    points_flat = points.reshape(-1, 3)
    labels_flat = labels.reshape(-1)
    confidence_flat = confidence.reshape(-1)
    colors_flat = colors.reshape(-1, 3)
    finite_confidence = confidence_flat[np.isfinite(confidence_flat)]
    if len(finite_confidence) == 0:
        print("error: no finite confidence values were found", file=sys.stderr)
        return 1
    confidence_threshold = (
        0.0
        if args.confidence_percentile == 0.0
        else float(np.percentile(finite_confidence, args.confidence_percentile))
    )
    base_valid = (
        np.isfinite(points_flat).all(axis=1)
        & np.isfinite(confidence_flat)
        & (confidence_flat >= confidence_threshold)
        & (confidence_flat > 1e-5)
    )

    manifest_rows = []
    skipped_small = 0
    for label in sorted(int(value) for value in np.unique(labels_flat) if value >= 0):
        indices = np.flatnonzero(base_valid & (labels_flat == label))
        original_count = len(indices)
        if original_count < args.min_points:
            skipped_small += 1
            continue
        if args.max_points and original_count > args.max_points:
            rng = np.random.default_rng(args.seed + label)
            indices = np.sort(rng.choice(indices, size=args.max_points, replace=False))

        instance_points = points_flat[indices]
        if alignment is not None:
            instance_points = trimesh.transform_points(instance_points, alignment)
        instance_colors = colors_flat[indices]
        output_path = output_dir / f"instance_{label:04d}.ply"
        point_count = export_colored_point_cloud(
            instance_points, instance_colors, output_path
        )
        manifest_rows.append(
            {
                "label_id": label,
                "valid_points": original_count,
                "exported_points": point_count,
                "file": output_path.name,
            }
        )
        print(
            f"Instance {label:4d}: {point_count:8d} points -> {output_path.name}"
        )

    if not manifest_rows:
        print(
            "error: no instances passed the confidence and minimum-point filters",
            file=sys.stderr,
        )
        return 1

    manifest_path = output_dir / "instances.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("label_id", "valid_points", "exported_points", "file")
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Exported instances: {len(manifest_rows)}")
    print(f"Skipped small:      {skipped_small}")
    print(f"Confidence cutoff: {confidence_threshold:.6f}")
    print(f"Manifest:           {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
