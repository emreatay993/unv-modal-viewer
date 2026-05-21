from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ModalModel, ModeShape
from .state import MacOptions, ModeNormalization


@dataclass(frozen=True, slots=True)
class NodePair:
    primary_label: int
    comparison_label: int
    distance: float = 0.0


def pair_nodes_by_id(primary: ModalModel, comparison: ModalModel) -> list[NodePair]:
    shared = sorted(set(primary.nodes).intersection(comparison.nodes))
    return [NodePair(label, label, 0.0) for label in shared]


def pair_nodes_by_nearest(primary: ModalModel, comparison: ModalModel, tolerance: float) -> list[NodePair]:
    tolerance = float(tolerance)
    pairs: list[NodePair] = []
    used_comparison: set[int] = set()
    comparison_labels = comparison.node_labels
    if not primary.nodes or not comparison_labels:
        return pairs

    comparison_points = np.vstack([comparison.nodes[label].coordinates for label in comparison_labels])
    for primary_label in primary.node_labels:
        point = primary.nodes[primary_label].coordinates
        distances = np.linalg.norm(comparison_points - point, axis=1)
        for index in np.argsort(distances):
            comparison_label = comparison_labels[int(index)]
            distance = float(distances[int(index)])
            if comparison_label in used_comparison:
                continue
            if distance <= tolerance:
                pairs.append(NodePair(primary_label, comparison_label, distance))
                used_comparison.add(comparison_label)
            break
    return pairs


def pair_nodes_for_mac(primary: ModalModel, comparison: ModalModel, options: MacOptions) -> list[NodePair]:
    pairs = pair_nodes_by_id(primary, comparison)
    if not options.use_nearest_fallback:
        return pairs

    primary_paired = {pair.primary_label for pair in pairs}
    comparison_paired = {pair.comparison_label for pair in pairs}
    primary_unmatched = _subset_model(primary, [label for label in primary.node_labels if label not in primary_paired])
    comparison_unmatched = _subset_model(
        comparison,
        [label for label in comparison.node_labels if label not in comparison_paired],
    )
    return [*pairs, *pair_nodes_by_nearest(primary_unmatched, comparison_unmatched, options.nearest_tolerance)]


def normalized_mode_vectors_for_nodes(
    model: ModalModel,
    mode: ModeShape | None,
    normalization: str = ModeNormalization.RAW,
    component: str = "Magnitude",
) -> tuple[np.ndarray, np.ndarray]:
    labels = model.node_labels
    vectors = np.zeros((len(labels), 3), dtype=float)
    if mode is None:
        return vectors, np.zeros(len(labels), dtype=float)

    for row, label in enumerate(labels):
        values = _mode_values(mode, label)
        if values.size:
            real_values = np.real(values)
            vectors[row, : min(3, real_values.size)] = real_values[:3]

    vectors = _normalize_vectors(vectors, mode, normalization)
    scalars = _scalars_from_vectors(vectors, mode, component)
    return vectors, scalars


def compute_mac_matrix(
    row_modes: list[ModeShape],
    column_modes: list[ModeShape],
    pairs: list[NodePair],
    components: str = "XYZ",
) -> np.ndarray:
    matrix = np.zeros((len(row_modes), len(column_modes)), dtype=float)
    for row, row_mode in enumerate(row_modes):
        left = _flatten_mode(row_mode, [pair.primary_label for pair in pairs], components)
        for column, column_mode in enumerate(column_modes):
            right = _flatten_mode(column_mode, [pair.comparison_label for pair in pairs], components)
            matrix[row, column] = _mac(left, right)
    return matrix


def best_mac_matches(matrix: np.ndarray) -> list[tuple[int, int, float]]:
    if matrix.size == 0:
        return []
    return [(row, int(np.argmax(matrix[row])), float(np.max(matrix[row]))) for row in range(matrix.shape[0])]


def mode_label(mode: ModeShape, index: int) -> str:
    mode_no = mode.mode_number if mode.mode_number is not None else index + 1
    freq = "" if mode.frequency_hz is None else f" {mode.frequency_hz:.4g} Hz"
    return f"Mode {mode_no}{freq}"


def _normalize_vectors(vectors: np.ndarray, mode: ModeShape, normalization: str) -> np.ndarray:
    out = vectors.copy()
    if normalization == ModeNormalization.MAX_DISPLACEMENT:
        magnitudes = np.linalg.norm(out, axis=1)
        maximum = float(np.max(magnitudes)) if magnitudes.size else 0.0
        if maximum > 0.0:
            out /= maximum
    elif normalization == ModeNormalization.UNIT_MODAL_MASS:
        if mode.modal_mass is not None and mode.modal_mass > 0.0:
            out /= float(np.sqrt(mode.modal_mass))
    return out


def _scalars_from_vectors(vectors: np.ndarray, mode: ModeShape, component: str) -> np.ndarray:
    component_index = {"X": 0, "Y": 1, "Z": 2, "Rx": 3, "Ry": 4, "Rz": 5}.get(component)
    if component_index is None:
        return np.linalg.norm(vectors, axis=1)
    if component_index >= vectors.shape[1]:
        return np.zeros(vectors.shape[0], dtype=float)
    return vectors[:, component_index]


def _flatten_mode(mode: ModeShape, labels: list[int], components: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for label in labels:
        values = _mode_values(mode, label)
        indices = _component_indices(components, values.size)
        if indices:
            chunks.append(values[indices])
    if not chunks:
        return np.empty(0, dtype=complex)
    return np.concatenate(chunks).astype(complex)


def _mode_values(mode: ModeShape, node_label: int) -> np.ndarray:
    raw = mode.node_values.get(node_label)
    if raw is None:
        return np.empty(0, dtype=complex)
    values = np.asarray(raw, dtype=float)
    if mode.data_type in (5, 6):
        return values[0::2] + 1j * values[1::2]
    return values.astype(complex)


def _component_indices(components: str, size: int) -> list[int]:
    if components == "X":
        candidates = [0]
    elif components == "Y":
        candidates = [1]
    elif components == "Z":
        candidates = [2]
    elif components == "RxRyRz":
        candidates = [3, 4, 5]
    elif components == "All available":
        candidates = list(range(size))
    else:
        candidates = [0, 1, 2]
    return [index for index in candidates if index < size]


def _mac(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    numerator = abs(np.vdot(left, right)) ** 2
    denominator = float(np.real(np.vdot(left, left)) * np.real(np.vdot(right, right)))
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _subset_model(model: ModalModel, labels: list[int]) -> ModalModel:
    return ModalModel(path=model.path, blocks=[], nodes={label: model.nodes[label] for label in labels})
