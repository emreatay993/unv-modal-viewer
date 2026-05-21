from __future__ import annotations

import numpy as np
import pytest

from unv_modal_viewer.modal_analysis import (
    compute_mac_matrix,
    normalized_mode_vectors_for_nodes,
    pair_nodes_by_id,
    pair_nodes_by_nearest,
)
from unv_modal_viewer.model import ModalModel, ModeShape, Node
from unv_modal_viewer.state import ModeNormalization


def _model(offset: float = 0.0) -> ModalModel:
    return ModalModel(
        path=None,
        blocks=[],
        nodes={
            1: Node(1, np.array([0.0 + offset, 0.0, 0.0])),
            2: Node(2, np.array([1.0 + offset, 0.0, 0.0])),
        },
    )


def _mode(number: int, values: dict[int, list[float]], mass: float | None = None) -> ModeShape:
    return ModeShape(
        name=f"Mode {number}",
        source_dataset=55,
        block_index=number,
        mode_number=number,
        frequency_hz=float(number),
        modal_mass=mass,
        viscous_damping=None,
        hysteretic_damping=None,
        data_characteristic=2,
        result_type=8,
        data_type=2,
        ndv=3,
        node_values={label: np.array(value, dtype=float) for label, value in values.items()},
    )


def test_mode_normalization_max_and_modal_mass() -> None:
    model = _model()
    mode = _mode(1, {1: [2.0, 0.0, 0.0], 2: [0.0, 4.0, 0.0]}, mass=4.0)

    vectors, magnitudes = normalized_mode_vectors_for_nodes(model, mode, ModeNormalization.MAX_DISPLACEMENT)
    mass_vectors, _ = normalized_mode_vectors_for_nodes(model, mode, ModeNormalization.UNIT_MODAL_MASS)

    assert magnitudes.tolist() == pytest.approx([0.5, 1.0])
    assert vectors[1].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert mass_vectors[0].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_mac_matrix_component_filters_and_pairing() -> None:
    model = _model()
    mode_x = _mode(1, {1: [1.0, 0.0, 0.0], 2: [1.0, 0.0, 0.0]})
    mode_y = _mode(2, {1: [0.0, 1.0, 0.0], 2: [0.0, 1.0, 0.0]})
    pairs = pair_nodes_by_id(model, model)

    matrix = compute_mac_matrix([mode_x, mode_y], [mode_x, mode_y], pairs, "XYZ")
    x_only = compute_mac_matrix([mode_x], [mode_x], pairs, "X")

    np.testing.assert_allclose(matrix, np.eye(2), atol=1.0e-12)
    assert x_only[0, 0] == pytest.approx(1.0)


def test_nearest_pairing_uses_tolerance() -> None:
    primary = _model()
    comparison = _model(offset=0.01)

    assert pair_nodes_by_nearest(primary, comparison, tolerance=0.001) == []
    pairs = pair_nodes_by_nearest(primary, comparison, tolerance=0.02)

    assert [(pair.primary_label, pair.comparison_label) for pair in pairs] == [(1, 1), (2, 2)]
