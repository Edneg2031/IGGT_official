#!/usr/bin/env python3
"""Extract colored point data from a GLB scene and save it as PLY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the point cloud in an IGGT GLB scene to a vertex-colored "
            "PLY file. Camera meshes are excluded by default."
        )
    )
    parser.add_argument("input_glb", type=Path, help="input .glb file")
    parser.add_argument(
        "output_ply",
        type=Path,
        nargs="?",
        help="output .ply file (default: INPUT_GLB with a .ply suffix)",
    )
    parser.add_argument(
        "--include-mesh-vertices",
        action="store_true",
        help="also convert mesh vertices, including the camera models",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite OUTPUT_PLY if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input_glb.expanduser().resolve()
    output_path = (
        args.output_ply.expanduser().resolve()
        if args.output_ply is not None
        else input_path.with_suffix(".ply")
    )

    if input_path.suffix.lower() != ".glb":
        print(f"error: input must be a .glb file: {input_path}", file=sys.stderr)
        return 2
    if not input_path.is_file():
        print(f"error: input file does not exist: {input_path}", file=sys.stderr)
        return 2
    if output_path.suffix.lower() != ".ply":
        print(f"error: output must use the .ply suffix: {output_path}", file=sys.stderr)
        return 2
    if output_path.exists() and not args.overwrite:
        print(
            f"error: output already exists: {output_path}\n"
            "Pass --overwrite to replace it.",
            file=sys.stderr,
        )
        return 1

    try:
        import trimesh
        from iggt.utils.pointcloud_io import export_scene_point_cloud
    except ImportError as exc:
        print(
            f"error: missing conversion dependency: {exc}\n"
            "Run this script in the IGGT environment.",
            file=sys.stderr,
        )
        return 1

    scene = trimesh.load_scene(str(input_path), process=False)
    try:
        stats = export_scene_point_cloud(
            scene,
            output_path,
            include_mesh_vertices=args.include_mesh_vertices,
        )
    except ValueError as exc:
        print(
            f"error: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Input:             {input_path}")
    print(f"Point geometries:  {stats['geometry_count']}")
    print(f"Skipped meshes:    {stats['skipped_mesh_count']}")
    print(f"Points written:    {stats['point_count']}")
    print(f"Non-finite removed:{stats['removed_nonfinite']:>8}")
    print(f"Output:            {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
