from __future__ import annotations

import numpy as np
import pyvista as pv

from .model import Element, ModalModel, ModeShape
from .transforms import mode_vectors_for_nodes


def point_cloud(model: ModalModel, points: np.ndarray | None = None) -> pv.PolyData:
    labels = model.node_labels
    coords = model.points if points is None else np.asarray(points, dtype=float)
    mesh = pv.PolyData(coords)
    if len(labels) == len(coords):
        mesh.point_data["node_id"] = np.array(labels, dtype=int)
    return mesh


def element_surface(model: ModalModel, points: np.ndarray | None = None) -> pv.PolyData | None:
    labels = model.node_labels
    index = {label: i for i, label in enumerate(labels)}
    coords = model.points if points is None else np.asarray(points, dtype=float)
    faces: list[int] = []

    for element in model.elements:
        node_indices = [index[label] for label in element.node_labels if label in index]
        if len(node_indices) >= 3:
            faces.extend([len(node_indices), *node_indices])

    if not faces:
        return None
    mesh = pv.PolyData(coords, np.array(faces, dtype=int))
    mesh.point_data["node_id"] = np.array(labels, dtype=int)
    return mesh


def trace_line_mesh(model: ModalModel, points: np.ndarray | None = None) -> pv.PolyData | None:
    labels = model.node_labels
    index = {label: i for i, label in enumerate(labels)}
    coords = model.points if points is None else np.asarray(points, dtype=float)
    lines: list[int] = []

    for trace in model.trace_lines:
        node_indices = [index[label] for label in trace.node_labels if label in index]
        if len(node_indices) >= 2:
            lines.extend([len(node_indices), *node_indices])

    if not lines:
        return None
    mesh = pv.PolyData(coords)
    mesh.lines = np.array(lines, dtype=int)
    mesh.point_data["node_id"] = np.array(labels, dtype=int)
    return mesh


def generated_surface(model: ModalModel, points: np.ndarray | None = None) -> pv.PolyData | None:
    labels = model.node_labels
    coords = model.points if points is None else np.asarray(points, dtype=float)
    if coords.shape[0] < 3:
        return None

    centered = coords - coords.mean(axis=0)
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if singular_values[1] < 1.0e-12:
            return None
        basis = vh[:2].T
        plane_points = centered @ basis
        plane_mesh = pv.PolyData(np.c_[plane_points, np.zeros(len(plane_points))])
        tri = plane_mesh.delaunay_2d()
        tri.points = coords.copy()
        tri.point_data["node_id"] = np.array(labels, dtype=int)
        return tri
    except Exception:
        return None


def deformed_points(
    model: ModalModel,
    mode: ModeShape | None,
    scale: float,
    component: str = "Magnitude",
    base_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = model.points if base_points is None else np.asarray(base_points, dtype=float)
    if mode is None:
        return points, np.zeros(points.shape[0], dtype=float)
    vectors, scalars = mode_vectors_for_nodes(model, mode, component)
    return points + vectors * float(scale), scalars


def supported_surface_elements(elements: list[Element]) -> int:
    return sum(1 for element in elements if len(element.node_labels) >= 3)

