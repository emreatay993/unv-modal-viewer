from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pytest

from unv_modal_viewer.io import export_unv, load_unv
from unv_modal_viewer.model import TransformSpec
from unv_modal_viewer.visualization import generated_surface

from .fixtures import generated_modal_unv, write_generated_modal_unv


def test_load_generated_modal_unv_reads_modal_test_datasets(tmp_path: Path) -> None:
    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    model = load_unv(path)

    assert model.units is not None
    assert model.units.code == 1
    assert len(model.nodes) == 4
    assert len(model.elements) == 1
    assert len(model.trace_lines) == 1
    assert len(model.coordinate_systems) == 1
    assert len(model.modes) == 2
    assert model.modes[0].source_dataset == 55
    assert model.modes[0].mode_number == 7
    assert model.modes[0].frequency_hz == pytest.approx(12.5)
    assert model.modes[0].viscous_damping == pytest.approx(0.03)
    assert model.modes[1].source_dataset == 2414
    assert model.modes[1].mode_number == 8
    assert model.modes[1].frequency_hz == pytest.approx(15.0)
    assert model.metadata["dataset_counts"][9999] == 1


def test_export_rewrites_coordinates_and_preserves_unknown_blocks(tmp_path: Path) -> None:
    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    model = load_unv(path)
    out = tmp_path / "modified.unv"
    spec = TransformSpec(scale=np.array([2.0, 2.0, 2.0]))

    export_unv(model, out, spec, transform_vectors=False)
    exported = out.read_text(encoding="latin-1")
    reloaded = load_unv(out)

    assert "UNKNOWN_PAYLOAD_SHOULD_STAY_BYTE_FOR_BYTE" in exported
    assert reloaded.nodes[2].coordinates.tolist() == pytest.approx([2.0, 0.0, 0.0])
    assert reloaded.modes[0].node_values[1].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_export_can_transform_dataset_55_and_2414_vectors(tmp_path: Path) -> None:
    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    model = load_unv(path)
    out = tmp_path / "modified_vectors.unv"
    spec = TransformSpec(scale=np.array([2.0, 3.0, 4.0]))

    export_unv(model, out, spec, transform_vectors=True)
    reloaded = load_unv(out)

    assert reloaded.modes[0].node_values[1].tolist() == pytest.approx([2.0, 0.0, 0.0])
    assert reloaded.modes[0].node_values[2].tolist() == pytest.approx([0.0, 6.0, 0.0])
    assert reloaded.modes[1].node_values[3].tolist() == pytest.approx([0.0, 0.0, 8.0])


def test_coordinate_system_alignment_uses_origin_and_rotation(tmp_path: Path) -> None:
    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    model = load_unv(path)
    cs = model.coordinate_systems[100]
    out = tmp_path / "aligned.unv"
    spec = TransformSpec(cs_rotation=cs.rotation, cs_origin=cs.origin)

    export_unv(model, out, spec)
    reloaded = load_unv(out)

    assert reloaded.nodes[1].coordinates.tolist() == pytest.approx([-2.0, 1.0, -3.0])


def test_generated_triangulated_surface_for_modal_points(tmp_path: Path) -> None:
    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    model = load_unv(path)

    surface = generated_surface(model)

    assert surface is not None
    assert surface.n_points == 4
    assert surface.n_cells >= 1


def test_public_pyuff_beam_modal_test_fixture_loads(tmp_path: Path) -> None:
    url = "https://raw.githubusercontent.com/ladisk/pyuff/main/data/beam.uff"
    path = tmp_path / "beam_as_modal_test.unv"
    try:
        urlretrieve(url, path)
    except Exception as exc:  # pragma: no cover - network dependent skip
        pytest.skip(f"public modal-test fixture unavailable: {exc}")

    model = load_unv(path)

    assert model.metadata["dataset_counts"][164] == 1
    assert model.metadata["dataset_counts"][2411] == 1
    assert model.metadata["dataset_counts"][58] == 3
    assert len(model.nodes) == 10
    assert len(model.functions) == 3
    assert model.units is not None
    assert model.units.code == 9


def test_public_pyuff_2414_mode_shape_fixture_loads(tmp_path: Path) -> None:
    url = "https://raw.githubusercontent.com/ladisk/pyuff/main/data/2411%20and%202414.uff"
    path = tmp_path / "public_2411_2414_modes.uff"
    try:
        urlretrieve(url, path)
    except Exception as exc:  # pragma: no cover - network dependent skip
        pytest.skip(f"public 2414 modal fixture unavailable: {exc}")

    model = load_unv(path)

    assert model.metadata["dataset_counts"][2411] == 1
    assert model.metadata["dataset_counts"][2412] >= 1
    assert model.metadata["dataset_counts"][2414] >= 1
    assert len(model.nodes) > 0
    assert len(model.elements) > 0
    assert len(model.modes) >= 1
    assert any(mode.source_dataset == 2414 and mode.node_values for mode in model.modes)


def test_public_pyuff_55_mode_shape_fixture_loads(tmp_path: Path) -> None:
    url = "https://raw.githubusercontent.com/ladisk/pyuff/main/data/uff55_translation.uff"
    path = tmp_path / "public_55_modes.uff"
    try:
        urlretrieve(url, path)
    except Exception as exc:  # pragma: no cover - network dependent skip
        pytest.skip(f"public dataset 55 modal fixture unavailable: {exc}")

    model = load_unv(path)

    assert model.metadata["dataset_counts"][55] == 3
    assert len(model.modes) == 3
    assert [mode.frequency_hz for mode in model.modes] == pytest.approx([10.0, 12.0, 13.0])
    assert all(mode.source_dataset == 55 for mode in model.modes)
    assert all(mode.node_values for mode in model.modes)


def test_fixture_contains_unknown_payload_once() -> None:
    assert generated_modal_unv().count("UNKNOWN_PAYLOAD_SHOULD_STAY_BYTE_FOR_BYTE") == 1
