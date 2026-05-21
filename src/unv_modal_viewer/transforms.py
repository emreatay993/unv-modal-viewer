from __future__ import annotations

import numpy as np

from .model import ModalModel, ModeShape, TransformSpec


def normalized_axes_from_rows(rows: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rows, dtype=float).reshape(3, 3)
    q, _ = np.linalg.qr(matrix.T)
    return q.T


def apply_transform(points: np.ndarray, spec: TransformSpec) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    scaled = pts * np.asarray(spec.scale, dtype=float)
    shifted = scaled + np.asarray(spec.translation, dtype=float)
    centered = shifted - np.asarray(spec.cs_origin, dtype=float)
    return centered @ np.asarray(spec.cs_rotation, dtype=float)


def transformed_node_coordinates(model: ModalModel, spec: TransformSpec) -> dict[int, np.ndarray]:
    labels = model.node_labels
    if not labels:
        return {}
    transformed = apply_transform(model.points, spec)
    return {label: transformed[i] for i, label in enumerate(labels)}


def transform_vector_values(values: np.ndarray, data_type: int, spec: TransformSpec) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if arr.size < 3:
        return arr

    rotation = np.asarray(spec.cs_rotation, dtype=float)
    scale = np.asarray(spec.scale, dtype=float)

    if data_type in (5, 6):
        real = arr[0::2].copy()
        imag = arr[1::2].copy()
        real = _transform_real_vector(real, rotation, scale)
        imag = _transform_real_vector(imag, rotation, scale)
        out = arr.copy()
        out[0::2] = real
        out[1::2] = imag
        return out

    return _transform_real_vector(arr, rotation, scale)


def transformed_mode_shape(mode: ModeShape, spec: TransformSpec) -> dict[int, np.ndarray]:
    return {
        node_label: transform_vector_values(values, mode.data_type, spec)
        for node_label, values in mode.node_values.items()
    }


def mode_vectors_for_nodes(
    model: ModalModel,
    mode: ModeShape,
    component: str = "Magnitude",
) -> tuple[np.ndarray, np.ndarray]:
    labels = model.node_labels
    vectors = np.zeros((len(labels), 3), dtype=float)
    scalars = np.zeros(len(labels), dtype=float)
    component_index = _component_index(component)

    for i, label in enumerate(labels):
        raw = mode.node_values.get(label)
        if raw is None or raw.size == 0:
            continue
        values = _real_components(raw, mode.data_type)
        if values.size >= 3:
            vectors[i] = values[:3]
        elif values.size > 0:
            vectors[i, 0] = values[0]
        if component_index is None:
            scalars[i] = float(np.linalg.norm(vectors[i]))
        elif component_index < values.size:
            scalars[i] = float(values[component_index])
    return vectors, scalars


def _real_components(values: np.ndarray, data_type: int) -> np.ndarray:
    if data_type in (5, 6):
        return np.asarray(values, dtype=float)[0::2]
    return np.asarray(values, dtype=float)


def _component_index(component: str) -> int | None:
    if component == "Magnitude":
        return None
    names = {"X": 0, "Y": 1, "Z": 2, "Rx": 3, "Ry": 4, "Rz": 5}
    return names.get(component)


def _transform_real_vector(values: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = values.copy()
    if out.size >= 3:
        out[:3] = (out[:3] * scale) @ rotation
    if out.size >= 6:
        out[3:6] = out[3:6] @ rotation
    return out

