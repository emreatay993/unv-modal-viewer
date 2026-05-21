from __future__ import annotations

import numpy as np
import pytest

from unv_modal_viewer.model import ModalModel, ModeShape, Node, TransformSpec
from unv_modal_viewer.transforms import (
    apply_transform,
    euler_degrees_from_rotation_matrix,
    mode_vectors_for_nodes,
    normalized_axes_from_rows,
    rotation_matrix_from_euler_degrees,
    transform_vector_values,
)
from unv_modal_viewer.visualization import generated_surface


ROT_Z_ROW_VECTOR = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def test_apply_transform_order_is_scale_then_translation_then_cs_alignment() -> None:
    points = np.array([[1.0, 2.0, 3.0]])
    spec = TransformSpec(
        scale=np.array([2.0, 3.0, 4.0]),
        translation=np.array([10.0, 20.0, 30.0]),
        cs_origin=np.array([1.0, 2.0, 3.0]),
        cs_rotation=ROT_Z_ROW_VECTOR,
    )

    transformed = apply_transform(points, spec)

    assert transformed[0].tolist() == pytest.approx([24.0, -11.0, 39.0])


def test_6dof_vector_transform_scales_translations_but_not_rotations() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    spec = TransformSpec(scale=np.array([2.0, 3.0, 4.0]), cs_rotation=ROT_Z_ROW_VECTOR)

    transformed = transform_vector_values(values, data_type=2, spec=spec)

    assert transformed.tolist() == pytest.approx([6.0, -2.0, 12.0, 5.0, -4.0, 6.0])


def test_complex_vector_transform_keeps_real_imaginary_interleaving() -> None:
    values = np.array([1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
    spec = TransformSpec(scale=np.array([2.0, 3.0, 4.0]), cs_rotation=ROT_Z_ROW_VECTOR)

    transformed = transform_vector_values(values, data_type=5, spec=spec)

    assert transformed.tolist() == pytest.approx([6.0, 60.0, -2.0, -20.0, 12.0, 120.0])


def test_mode_vectors_use_real_components_for_complex_modes() -> None:
    model = ModalModel(
        path=None,
        blocks=[],
        nodes={
            10: Node(10, np.array([0.0, 0.0, 0.0])),
            20: Node(20, np.array([1.0, 0.0, 0.0])),
        },
    )
    mode = ModeShape(
        name="complex",
        source_dataset=55,
        block_index=0,
        mode_number=1,
        frequency_hz=1.0,
        modal_mass=None,
        viscous_damping=None,
        hysteretic_damping=None,
        data_characteristic=2,
        result_type=8,
        data_type=5,
        ndv=3,
        node_values={
            10: np.array([3.0, 30.0, 4.0, 40.0, 0.0, 50.0]),
            20: np.array([0.0, 10.0, 0.0, 20.0, 5.0, 30.0]),
        },
    )

    vectors, magnitudes = mode_vectors_for_nodes(model, mode, "Magnitude")
    _, z_values = mode_vectors_for_nodes(model, mode, "Z")

    np.testing.assert_allclose(vectors, np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]))
    assert magnitudes.tolist() == pytest.approx([5.0, 5.0])
    assert z_values.tolist() == pytest.approx([0.0, 5.0])


def test_normalized_axes_from_rows_returns_orthonormal_basis() -> None:
    axes = normalized_axes_from_rows(np.array([[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 4.0]]))

    assert axes @ axes.T == pytest.approx(np.eye(3))


def test_rotation_angles_match_existing_row_vector_matrix_convention() -> None:
    rotation = rotation_matrix_from_euler_degrees(0.0, 0.0, 90.0)

    np.testing.assert_allclose(rotation, ROT_Z_ROW_VECTOR, atol=1.0e-12)


def test_rotation_angles_round_trip_through_matrix() -> None:
    rotation = rotation_matrix_from_euler_degrees(25.0, -10.0, 40.0)
    angles = euler_degrees_from_rotation_matrix(rotation)
    rebuilt = rotation_matrix_from_euler_degrees(*angles)

    np.testing.assert_allclose(rebuilt, rotation, atol=1.0e-12)


def test_generated_surface_returns_none_for_collinear_points() -> None:
    model = ModalModel(
        path=None,
        blocks=[],
        nodes={
            1: Node(1, np.array([0.0, 0.0, 0.0])),
            2: Node(2, np.array([1.0, 0.0, 0.0])),
            3: Node(3, np.array([2.0, 0.0, 0.0])),
        },
    )

    assert generated_surface(model) is None
