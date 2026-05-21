from __future__ import annotations

import numpy as np
import pytest

from .fixtures import write_generated_modal_unv


def _settings(tmp_path):
    from qtpy.QtCore import QSettings

    from unv_modal_viewer.settings import AppSettings

    ini_format = getattr(QSettings, "IniFormat", None) or QSettings.Format.IniFormat
    return AppSettings(QSettings(str(tmp_path / "settings.ini"), ini_format))


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

    window = MainWindow(path, settings=_settings(tmp_path))
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

    window = MainWindow(path, settings=_settings(tmp_path))
    try:
        window.rot_z.setValue(90.0)
        np.testing.assert_allclose(
            window.current_transform().cs_rotation,
            rotation_matrix_from_euler_degrees(0.0, 0.0, 90.0),
            atol=1.0e-12,
        )
    finally:
        window.close()


def test_animation_tick_updates_meshes_without_rebuilding_scene(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PyQt6")
    from qtpy.QtWidgets import QApplication

    import unv_modal_viewer.gui as gui
    from unv_modal_viewer.gui import MainWindow, set_fusion_theme

    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    app = QApplication.instance() or QApplication([])
    set_fusion_theme(app)

    window = MainWindow(path, settings=_settings(tmp_path))
    try:
        window.show_surface.setChecked(False)
        window.generate_surface.setChecked(True)
        window.refresh_scene(reset_camera=False)
        assert "surface" in window._scene_meshes
        assert window.animation_time.isHidden()
        assert window.animation_fps.isHidden()
        assert not window.export_animation_button.isEnabled()

        window.animation_time.setValue(2.0)
        window.animation_fps.setValue(20)
        clear_count = window.plotter.clear_count
        mesh_call_count = len(window.plotter.mesh_calls)
        render_count = window.plotter.render_count
        before_points = np.asarray(window._scene_meshes["surface"].points).copy()
        times = iter([100.0, 100.25])
        monkeypatch.setattr(gui.time, "perf_counter", lambda: next(times))

        window._animation_toggled(True)
        assert not window.animation_time.isHidden()
        assert not window.animation_fps.isHidden()
        assert window.export_animation_button.isEnabled()
        window._animation_tick()

        after_points = np.asarray(window._scene_meshes["surface"].points).copy()
        assert window.animation_timer.interval() == 50
        assert window._phase == pytest.approx(np.sin(np.pi / 4.0))
        assert window.plotter.clear_count == clear_count
        assert len(window.plotter.mesh_calls) == mesh_call_count
        assert window.plotter.render_count > render_count
        assert not np.allclose(before_points, after_points)

        window._animation_toggled(False)
        assert window.animation_time.isHidden()
        assert window.animation_fps.isHidden()
        assert not window.export_animation_button.isEnabled()
    finally:
        window.animation_timer.stop()
        window.close()


def test_left_panel_sections_are_collapsible(tmp_path) -> None:
    pytest.importorskip("PyQt6")
    from qtpy.QtWidgets import QApplication

    from unv_modal_viewer.gui import MainWindow, set_fusion_theme

    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    app = QApplication.instance() or QApplication([])
    set_fusion_theme(app)

    window = MainWindow(path, settings=_settings(tmp_path))
    try:
        for section in window._sections().values():
            assert not section.is_expanded()
            assert section.content.isHidden()

        window.transform_section.set_expanded(True)
        assert window.transform_section.is_expanded()
        assert not window.transform_section.content.isHidden()
    finally:
        window.close()


def test_sample_action_and_recent_files_are_wired(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PyQt6")
    from qtpy.QtWidgets import QApplication

    import unv_modal_viewer.gui as gui
    from unv_modal_viewer.gui import MainWindow, set_fusion_theme

    path = write_generated_modal_unv(tmp_path / "sample.unv")
    monkeypatch.setattr(gui, "ensure_sample_file", lambda name: path)
    app = QApplication.instance() or QApplication([])
    set_fusion_theme(app)

    window = MainWindow(settings=_settings(tmp_path))
    try:
        window.load_sample_button.click()
        assert window.model is not None
        assert window.recent_combo.count() == 1
        assert window.recent_combo.currentData() == str(path.resolve())
    finally:
        window.close()


def test_selection_hide_isolate_and_show_all(tmp_path) -> None:
    pytest.importorskip("PyQt6")
    from qtpy.QtWidgets import QApplication

    from unv_modal_viewer.gui import MainWindow, set_fusion_theme

    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    app = QApplication.instance() or QApplication([])
    set_fusion_theme(app)

    window = MainWindow(path, settings=_settings(tmp_path))
    try:
        window._select_point_index(0)
        assert window.selection.selected_node_ids == {1}
        window._hide_selected()
        assert 1 in window.selection.hidden_node_ids
        assert 1 not in window._current_labels

        window.selection.selected_node_ids = {2}
        window.isolate_selected.setChecked(True)
        assert window._current_labels == [2]

        window._show_all_hidden()
        assert not window.selection.hidden_node_ids
        assert len(window._current_labels) == 4
    finally:
        window.close()


def test_appearance_controls_update_render_options(tmp_path) -> None:
    pytest.importorskip("PyQt6")
    from qtpy.QtWidgets import QApplication

    from unv_modal_viewer.gui import MainWindow, set_fusion_theme

    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    app = QApplication.instance() or QApplication([])
    set_fusion_theme(app)

    window = MainWindow(path, settings=_settings(tmp_path))
    try:
        window.colormap_combo.setCurrentText("plasma")
        window.reverse_colormap.setChecked(True)
        window.legend_position.setCurrentText("Right")

        options = window.current_render_options()

        assert options.pyvista_colormap == "plasma_r"
        assert options.legend_position == "Right"
    finally:
        window.close()


def test_max_displacement_magnitude_uses_zero_to_one_legend_range(tmp_path) -> None:
    pytest.importorskip("PyQt6")
    from qtpy.QtWidgets import QApplication

    from unv_modal_viewer.gui import MainWindow, set_fusion_theme

    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    app = QApplication.instance() or QApplication([])
    set_fusion_theme(app)

    window = MainWindow(path, settings=_settings(tmp_path))
    try:
        window.scalar_auto.setChecked(True)
        window.normalization_combo.setCurrentText("Max displacement")
        window.component_combo.setCurrentText("Magnitude")
        window.refresh_scene(reset_camera=False)

        scalar_mesh_kwargs = [
            kwargs for _, kwargs in window.plotter.mesh_calls if kwargs.get("scalars") == "value"
        ]
        assert scalar_mesh_kwargs
        assert {kwargs.get("clim") for kwargs in scalar_mesh_kwargs} == {(0.0, 1.0)}

        window.component_combo.setCurrentText("Z")
        window.refresh_scene(reset_camera=False)

        z_scalar_mesh_kwargs = [
            kwargs for _, kwargs in window.plotter.mesh_calls if kwargs.get("scalars") == "value"
        ]
        assert z_scalar_mesh_kwargs
        assert {kwargs.get("clim") for kwargs in z_scalar_mesh_kwargs} == {None}
    finally:
        window.close()


def test_diagnostics_panel_and_overlay_state(tmp_path) -> None:
    pytest.importorskip("PyQt6")
    from qtpy.QtWidgets import QApplication

    from unv_modal_viewer.gui import MainWindow, set_fusion_theme

    path = write_generated_modal_unv(tmp_path / "modal_test.unv")
    overlay_path = write_generated_modal_unv(tmp_path / "overlay.unv")
    app = QApplication.instance() or QApplication([])
    set_fusion_theme(app)

    window = MainWindow(path, settings=_settings(tmp_path))
    try:
        window.model.diagnostics.append("diagnostic marker")
        window._update_diagnostics_panel()
        assert "diagnostic marker" in window.diagnostics_text.toPlainText()

        window._load_overlay_file(overlay_path)
        window.overlay_trans_x.setValue(5.0)
        window._overlay_controls_changed()
        assert window.overlay.model is not None
        assert window.current_overlay_transform().translation[0] == pytest.approx(5.0)
    finally:
        window.close()
