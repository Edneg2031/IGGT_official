"""Utilities for exporting colored point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def colors_to_uint8(colors: Any, vertex_count: int) -> np.ndarray:
    """Normalize RGB/RGBA colors to an ``(N, 3)`` uint8 array."""
    if colors is None:
        return np.full((vertex_count, 3), 255, dtype=np.uint8)

    colors_array = np.asarray(colors)
    if (
        colors_array.ndim != 2
        or len(colors_array) != vertex_count
        or colors_array.shape[1] < 3
    ):
        return np.full((vertex_count, 3), 255, dtype=np.uint8)

    colors_array = colors_array[:, :3]
    if (
        np.issubdtype(colors_array.dtype, np.floating)
        and colors_array.size
        and colors_array.max() <= 1.0
    ):
        colors_array = colors_array * 255.0
    return np.clip(colors_array, 0, 255).astype(np.uint8)


def geometry_colors(geometry: Any, vertex_count: int) -> np.ndarray:
    """Extract vertex colors from a trimesh geometry."""
    if hasattr(geometry, "colors"):
        candidate = np.asarray(geometry.colors)
        if candidate.ndim == 2 and len(candidate) == vertex_count:
            return colors_to_uint8(candidate, vertex_count)

    if hasattr(geometry, "visual"):
        candidate = np.asarray(
            getattr(geometry.visual, "vertex_colors", np.empty((0, 4)))
        )
        if candidate.ndim == 2 and len(candidate) == vertex_count:
            return colors_to_uint8(candidate, vertex_count)

    return colors_to_uint8(None, vertex_count)


def scene_to_point_cloud(
    scene: trimesh.Scene, include_mesh_vertices: bool = False
) -> tuple[trimesh.points.PointCloud, dict[str, int]]:
    """Extract transformed point geometry from a scene."""
    vertices_parts: list[np.ndarray] = []
    colors_parts: list[np.ndarray] = []
    geometry_count = 0
    skipped_mesh_count = 0

    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        is_point_cloud = isinstance(geometry, trimesh.points.PointCloud)
        is_faceless_mesh = (
            isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) == 0
        )

        if not (is_point_cloud or is_faceless_mesh) and not include_mesh_vertices:
            skipped_mesh_count += 1
            continue

        vertices = np.asarray(geometry.vertices)
        if vertices.size == 0:
            continue

        vertices = trimesh.transform_points(vertices, transform)
        vertices_parts.append(vertices)
        colors_parts.append(geometry_colors(geometry, len(vertices)))
        geometry_count += 1

    if not vertices_parts:
        raise ValueError(
            "No point-cloud geometry was found in the scene. "
            "Set include_mesh_vertices=True to convert all geometry vertices."
        )

    vertices = np.concatenate(vertices_parts, axis=0)
    colors = np.concatenate(colors_parts, axis=0)
    finite_mask = np.isfinite(vertices).all(axis=1)
    removed_nonfinite = int((~finite_mask).sum())
    vertices = vertices[finite_mask]
    colors = colors[finite_mask]

    if len(vertices) == 0:
        raise ValueError("All point coordinates in the scene are non-finite.")

    cloud = trimesh.points.PointCloud(vertices=vertices, colors=colors)
    stats = {
        "geometry_count": geometry_count,
        "skipped_mesh_count": skipped_mesh_count,
        "removed_nonfinite": removed_nonfinite,
        "point_count": len(vertices),
    }
    return cloud, stats


def export_scene_point_cloud(
    scene: trimesh.Scene,
    output_path: str | Path,
    include_mesh_vertices: bool = False,
) -> dict[str, int]:
    """Export only the point data in a scene to a vertex-colored PLY."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud, stats = scene_to_point_cloud(scene, include_mesh_vertices)
    cloud.export(str(output_path), file_type="ply")
    return stats


def export_colored_point_cloud(
    vertices: np.ndarray, colors: np.ndarray, output_path: str | Path
) -> int:
    """Export vertex and RGB arrays as a colored PLY point cloud."""
    vertices = np.asarray(vertices).reshape(-1, 3)
    colors = colors_to_uint8(np.asarray(colors).reshape(-1, 3), len(vertices))
    finite_mask = np.isfinite(vertices).all(axis=1)
    vertices = vertices[finite_mask]
    colors = colors[finite_mask]
    if len(vertices) == 0:
        raise ValueError("No finite points are available for export.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.points.PointCloud(vertices=vertices, colors=colors)
    cloud.export(str(output_path), file_type="ply")
    return len(vertices)
