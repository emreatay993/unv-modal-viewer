from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from unv_modal_viewer.exporters import (
    export_animation_media,
    export_mac_csv,
    export_modes_csv,
    export_nodes_csv,
    export_scene_vtk,
)
from unv_modal_viewer.io import load_unv
from unv_modal_viewer.model import TransformSpec

from .fixtures import write_generated_modal_unv


def test_export_nodes_and_modes_csv(tmp_path: Path) -> None:
    model = load_unv(write_generated_modal_unv(tmp_path / "modal.unv"))
    nodes_csv = tmp_path / "nodes.csv"
    modes_csv = tmp_path / "modes.csv"

    export_nodes_csv(model, nodes_csv, TransformSpec(scale=np.array([2.0, 2.0, 2.0])), node_labels=[1])
    export_modes_csv(model, modes_csv, modes=[model.modes[0]], node_labels=[1])

    with nodes_csv.open(newline="", encoding="utf-8") as handle:
        node_rows = list(csv.DictReader(handle))
    with modes_csv.open(newline="", encoding="utf-8") as handle:
        mode_rows = list(csv.DictReader(handle))

    assert node_rows[0]["node_id"] == "1"
    assert node_rows[0]["transformed_x"] == "0"
    assert mode_rows[0]["component"] == "X"
    assert mode_rows[0]["mode"] == "7"


def test_export_mac_csv(tmp_path: Path) -> None:
    model = load_unv(write_generated_modal_unv(tmp_path / "modal.unv"))
    path = tmp_path / "mac.csv"

    export_mac_csv(path, np.eye(2), model.modes, model.modes)

    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    assert rows[0][0] == "mode"
    assert rows[1][1] == "1"


def test_export_scene_vtk_contains_node_metadata(tmp_path: Path) -> None:
    pv = pytest.importorskip("pyvista")
    model = load_unv(write_generated_modal_unv(tmp_path / "modal.unv"))
    path = tmp_path / "scene.vtp"

    export_scene_vtk(
        model,
        path,
        model.points,
        np.arange(len(model.node_labels), dtype=float),
        selected_labels={1},
        hidden_labels={2},
    )

    mesh = pv.read(path)
    assert "node_id" in mesh.point_data
    assert "selected" in mesh.point_data
    assert "hidden" in mesh.point_data


def test_export_animation_media_uses_duration_fps_and_rgb_frames(tmp_path: Path) -> None:
    class FakeWriter:
        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def append_data(self, frame: np.ndarray) -> None:
            self.frames.append(frame)

    writer = FakeWriter()
    phases: list[float] = []

    def writer_factory(path: Path, fps: int) -> FakeWriter:
        assert path == tmp_path / "mode.mp4"
        assert fps == 4
        return writer

    def frame_source(phase: float) -> np.ndarray:
        phases.append(phase)
        return np.zeros((3, 5, 4), dtype=np.uint8)

    frame_count = export_animation_media(
        tmp_path / "mode.mp4",
        frame_source,
        duration_seconds=1.0,
        fps=4,
        writer_factory=writer_factory,
    )

    assert frame_count == 4
    assert phases == pytest.approx([0.0, 1.0, 0.0, -1.0], abs=1.0e-12)
    assert len(writer.frames) == 4
    assert writer.frames[0].shape == (3, 5, 3)
    assert writer.frames[0].dtype == np.uint8


def test_export_animation_media_rejects_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mp4"):
        export_animation_media(tmp_path / "mode.mov", lambda phase: np.zeros((2, 2, 3)), 1.0, 10)
