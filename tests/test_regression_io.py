from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unv_modal_viewer.io import export_unv, load_unv
from unv_modal_viewer.model import TransformSpec


def test_prefix_suffix_and_unknown_blocks_are_preserved_exactly(tmp_path: Path) -> None:
    source = (
        "HEADER LINE BEFORE DATASETS\n"
        "    -1\n"
        "  9999\n"
        "UNKNOWN LINE 1\n"
        "  -1 is data here, not a delimiter because it is not alone\n"
        "    -1\n"
        "TRAILING FOOTER AFTER DATASETS\n"
    )
    path = tmp_path / "unknown_only.unv"
    out = tmp_path / "roundtrip.unv"
    path.write_text(source, encoding="latin-1")

    export_unv(load_unv(path), out, TransformSpec.identity())

    assert out.read_text(encoding="latin-1") == source


def test_dataset_15_and_2411_duplicate_node_precedence(tmp_path: Path) -> None:
    path = tmp_path / "duplicate_nodes.unv"
    path.write_text(
        """    -1
    15
         1         0         0         7  9.00000E+00  9.00000E+00  9.00000E+00
    -1
    -1
  2411
         1         0         0         7
   1.0000000000000000D+00   2.0000000000000000D+00   3.0000000000000000D+00
    -1
    -1
    15
         1         0         0         7  8.00000E+00  8.00000E+00  8.00000E+00
    -1
""",
        encoding="latin-1",
    )

    model = load_unv(path)

    assert model.nodes[1].source_dataset == 2411
    assert model.nodes[1].coordinates.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_dataset_2412_beam_orientation_record_is_not_connectivity(tmp_path: Path) -> None:
    path = tmp_path / "beam_element.unv"
    path.write_text(
        """    -1
  2412
        50        21         0         0         8         2
        99         0         0
         1         2
    -1
""",
        encoding="latin-1",
    )

    model = load_unv(path)

    assert len(model.elements) == 1
    assert model.elements[0].descriptor == 21
    assert model.elements[0].node_labels == [1, 2]


def test_dataset_58_summary_parses_offline_modal_test_function(tmp_path: Path) -> None:
    path = tmp_path / "frf_summary.unv"
    path.write_text(
        """    -1
    58
FRF H1
Run
21-May-26 12:00:00
Load case
Acceleration / force
    4       101    1         0 NONE         3   -2 NONE         9    1
         5        12         1  0.00000E+00  2.50000E-01  0.00000E+00
        18    0    0    0 Frequency           Hz
        12    0    0    0 Acceleration        m/s2
         13    0    0    0 Force               N
         0    0    0    0 NONE                NONE
  1.00000E+00  0.00000E+00  2.00000E+00  0.00000E+00
    -1
""",
        encoding="latin-1",
    )

    model = load_unv(path)

    assert len(model.functions) == 1
    function = model.functions[0]
    assert function.description == "FRF H1"
    assert function.function_type == 4
    assert function.function_id == 101
    assert function.response_node == 3
    assert function.response_direction == -2
    assert function.reference_node == 9
    assert function.reference_direction == 1
    assert function.n_values == 12
    assert function.x_increment == pytest.approx(0.25)


def test_truncated_dataset_55_node_values_do_not_consume_next_node_label(tmp_path: Path) -> None:
    path = tmp_path / "truncated_55.unv"
    path.write_text(
        """    -1
    55
Mode
Run
Date
Case
Shape
         1         2         2         8         2         3
         2         4         1         1
  1.00000E+01  1.00000E+00  1.00000E-02  2.00000E-02
         1
  1.00000E+00  2.00000E+00
         2
  3.00000E+00  4.00000E+00  5.00000E+00
    -1
""",
        encoding="latin-1",
    )

    model = load_unv(path)

    assert len(model.modes) == 1
    assert sorted(model.modes[0].node_values) == [2]
    assert model.modes[0].node_values[2].tolist() == pytest.approx([3.0, 4.0, 5.0])


def test_dataset_2414_direct_normal_mode_parameters_and_label_fallback(tmp_path: Path) -> None:
    path = tmp_path / "direct_2414.unv"
    path.write_text(
        """    -1
  2414
        42
Direct parameter mode
         1
Model
Run
Date
Case
Shape
         1         2         2         8         2         3
         0         0         0         0         0         0         0         0
         0         0
  2.20000E+01  4.00000E+00  5.00000E-02  6.00000E-02  0.00000E+00  0.00000E+00
  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00
         7
  1.00000E+00  2.00000E+00  3.00000E+00
    -1
""",
        encoding="latin-1",
    )

    model = load_unv(path)

    assert len(model.modes) == 1
    mode = model.modes[0]
    assert mode.source_dataset == 2414
    assert mode.mode_number == 42
    assert mode.frequency_hz == pytest.approx(22.0)
    assert mode.modal_mass == pytest.approx(4.0)
    assert mode.viscous_damping == pytest.approx(0.05)
    assert mode.hysteretic_damping == pytest.approx(0.06)
    assert mode.node_values[7].tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_export_identity_keeps_dataset_55_block_text_when_vectors_not_requested(tmp_path: Path) -> None:
    source = """    -1
    55
Mode
Run
Date
Case
Shape
         1         2         2         8         2         3
         2         4         1         1
  1.00000E+01  1.00000E+00  1.00000E-02  2.00000E-02
         1
  1.00000E+00  2.00000E+00  3.00000E+00
    -1
"""
    path = tmp_path / "mode_55.unv"
    out = tmp_path / "mode_55_out.unv"
    path.write_text(source, encoding="latin-1")

    export_unv(load_unv(path), out, TransformSpec.identity(), transform_vectors=False)

    assert out.read_text(encoding="latin-1") == source

