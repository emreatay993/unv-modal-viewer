from __future__ import annotations

import numpy as np
import pytest

from .fixtures import write_generated_modal_unv


def test_qt_fusion_theme_can_be_applied() -> None:
    pytest.importorskip("PyQt6")
    from qtpy.QtWidgets import QApplication

    from unv_modal_viewer.gui import set_fusion_theme

    app = QApplication.instance() or QApplication([])
    set_fusion_theme(app)

    assert app.property("unv_modal_viewer_theme") == "fusion"


def test_main_window_loads_modal_fixture_offscreen(tmp_path) -> None:
    pytest.importorskip("PyQt6")
    from qtpy.QtWidgets import QApplication

    from unv_modal_viewer.gui import MainWindow, set_fusion_theme

    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    app = QApplication.instance() or QApplication([])
    set_fusion_theme(app)

    window = MainWindow(path)
    try:
        assert window.model is not None
        assert len(window.model.nodes) == 4
        assert window.mode_table.rowCount() == 2
    finally:
        window.close()


def test_main_window_uses_angle_rotation_controls(tmp_path) -> None:
    pytest.importorskip("PyQt6")
    from qtpy.QtWidgets import QApplication

    from unv_modal_viewer.gui import MainWindow, set_fusion_theme
    from unv_modal_viewer.transforms import rotation_matrix_from_euler_degrees

    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    app = QApplication.instance() or QApplication([])
    set_fusion_theme(app)

    window = MainWindow(path)
    try:
        window.rot_z.setValue(90.0)
        np.testing.assert_allclose(
            window.current_transform().cs_rotation,
            rotation_matrix_from_euler_degrees(0.0, 0.0, 90.0),
            atol=1.0e-12,
        )
    finally:
        window.close()
