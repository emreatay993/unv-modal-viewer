from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from qtpy.QtCore import QTimer, Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyleFactory,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from .exporters import (
    SUPPORTED_ANIMATION_EXTENSIONS,
    export_animation_media,
    export_mac_csv,
    export_modes_csv,
    export_nodes_csv,
    export_scene_vtk,
    export_screenshot,
)
from .io import SUPPORTED_DATASETS, export_unv, load_unv
from .modal_analysis import best_mac_matches, compute_mac_matrix, mode_label, pair_nodes_for_mac
from .model import CoordinateSystem, ModalModel, ModeShape, TransformSpec
from .samples import SAMPLES, ensure_sample_file
from .settings import AppSettings
from .state import MacOptions, ModeNormalization, OverlayState, RenderOptions, SelectionState, color_choices
from .transforms import (
    euler_degrees_from_rotation_matrix,
    rotation_matrix_from_euler_degrees,
    transformed_node_coordinates,
)
from .visualization import (
    deformed_points,
    element_surface,
    generated_surface,
    point_cloud,
    supported_surface_elements,
    trace_line_mesh,
)


class MainWindow(QMainWindow):
    def __init__(self, initial_path: str | Path | None = None, settings: AppSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle("UNV Modal Test Viewer")
        self.resize(1440, 900)

        self.settings = settings or AppSettings()
        self.model: ModalModel | None = None
        self.current_path: Path | None = None
        self.render_options = self.settings.load_render_options()
        self.selection = SelectionState()
        self.overlay = OverlayState()
        self._messages: list[str] = []
        self._mac_matrix = np.empty((0, 0), dtype=float)
        self._mac_row_modes: list[ModeShape] = []
        self._mac_column_modes: list[ModeShape] = []
        self._all_current_points = np.empty((0, 3), dtype=float)
        self._all_current_scalars = np.empty(0, dtype=float)
        self._all_current_labels: list[int] = []
        self._current_points = np.empty((0, 3), dtype=float)
        self._current_scalars = np.empty(0, dtype=float)
        self._current_labels: list[int] = []
        self._scene_meshes: dict[str, object] = {}
        self._phase = 1.0
        self._animation_started_at = 0.0
        self._hover_observer_installed = False
        self._point_picker = _make_point_picker()

        self._build_ui()
        self._connect_signals()
        self._install_hover_observer()
        self.refresh_scene(reset_camera=True)

        if initial_path:
            self.load_file(Path(initial_path))

    def _build_ui(self) -> None:
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self._build_left_panel())

        self.plotter = _NullPlotter(self) if os.getenv("UNV_MODAL_VIEWER_TEST_NO_VTK") else QtInteractor(self)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.splitter.addWidget(self.plotter)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([450, 990])
        self.setCentralWidget(self.splitter)
        self.statusBar().showMessage("Ready")
        self._restore_persisted_ui()

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("UNV Modal Test Viewer")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.file_section = _CollapsibleSection("File")
        file_layout = QVBoxLayout()
        self.file_section.set_content_layout(file_layout)
        self.path_label = QLabel("No file loaded")
        self.path_label.setWordWrap(True)
        file_layout.addWidget(self.path_label)
        file_buttons = QHBoxLayout()
        self.open_button = QPushButton("Open")
        self.open_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.export_button = QPushButton("Export")
        self.export_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.export_button.setEnabled(False)
        file_buttons.addWidget(self.open_button)
        file_buttons.addWidget(self.export_button)
        file_layout.addLayout(file_buttons)
        recent_buttons = QHBoxLayout()
        self.recent_combo = QComboBox()
        self.recent_combo.setMinimumWidth(180)
        self.open_recent_button = QPushButton("Open Recent")
        recent_buttons.addWidget(self.recent_combo, 1)
        recent_buttons.addWidget(self.open_recent_button)
        file_layout.addLayout(recent_buttons)
        sample_buttons = QHBoxLayout()
        self.sample_combo = QComboBox()
        self.sample_combo.addItems(list(SAMPLES))
        self.load_sample_button = QPushButton("Load Sample")
        sample_buttons.addWidget(self.sample_combo, 1)
        sample_buttons.addWidget(self.load_sample_button)
        file_layout.addLayout(sample_buttons)
        self.summary_label = QLabel("Datasets: none")
        self.summary_label.setWordWrap(True)
        file_layout.addWidget(self.summary_label)
        layout.addWidget(self.file_section)

        self.transform_section = _CollapsibleSection("Coordinate Transform")
        transform_layout = QFormLayout()
        self.transform_section.set_content_layout(transform_layout)
        transform_layout.setLabelAlignment(Qt.AlignLeft)
        self.scale_x = _double_box(1.0, minimum=-1.0e9, maximum=1.0e9, step=0.1)
        self.scale_y = _double_box(1.0, minimum=-1.0e9, maximum=1.0e9, step=0.1)
        self.scale_z = _double_box(1.0, minimum=-1.0e9, maximum=1.0e9, step=0.1)
        transform_layout.addRow("Scale X", self.scale_x)
        transform_layout.addRow("Scale Y", self.scale_y)
        transform_layout.addRow("Scale Z", self.scale_z)
        self.trans_x = _double_box(0.0)
        self.trans_y = _double_box(0.0)
        self.trans_z = _double_box(0.0)
        transform_layout.addRow("Translate X", self.trans_x)
        transform_layout.addRow("Translate Y", self.trans_y)
        transform_layout.addRow("Translate Z", self.trans_z)
        self.cs_combo = QComboBox()
        self.cs_combo.addItem("Global / Manual", None)
        transform_layout.addRow("Coordinate system", self.cs_combo)
        self.origin_x = _double_box(0.0)
        self.origin_y = _double_box(0.0)
        self.origin_z = _double_box(0.0)
        transform_layout.addRow("CS origin X", self.origin_x)
        transform_layout.addRow("CS origin Y", self.origin_y)
        transform_layout.addRow("CS origin Z", self.origin_z)
        layout.addWidget(self.transform_section)

        self.rotation_section = _CollapsibleSection("Rotation Angles")
        rotation_layout = QFormLayout()
        self.rotation_section.set_content_layout(rotation_layout)
        self.rot_x = _angle_box(0.0)
        self.rot_y = _angle_box(0.0)
        self.rot_z = _angle_box(0.0)
        self.rot_x.setToolTip("X rotation in degrees. Rotations are applied in X, then Y, then Z order.")
        self.rot_y.setToolTip("Y rotation in degrees. Rotations are applied in X, then Y, then Z order.")
        self.rot_z.setToolTip("Z rotation in degrees. Rotations are applied in X, then Y, then Z order.")
        rotation_layout.addRow("X angle", self.rot_x)
        rotation_layout.addRow("Y angle", self.rot_y)
        rotation_layout.addRow("Z angle", self.rot_z)
        layout.addWidget(self.rotation_section)

        self.view_section = _CollapsibleSection("View")
        view_layout = QFormLayout()
        self.view_section.set_content_layout(view_layout)
        self.show_points = QCheckBox()
        self.show_points.setChecked(True)
        self.show_surface = QCheckBox()
        self.show_surface.setChecked(True)
        self.generate_surface = QCheckBox()
        self.generate_surface.setChecked(True)
        self.show_traces = QCheckBox()
        self.show_traces.setChecked(True)
        view_layout.addRow("Points", self.show_points)
        view_layout.addRow("File topology", self.show_surface)
        view_layout.addRow("Generated surface", self.generate_surface)
        view_layout.addRow("Trace lines", self.show_traces)
        layout.addWidget(self.view_section)

        self.appearance_section = _CollapsibleSection("Appearance")
        appearance_layout = QFormLayout()
        self.appearance_section.set_content_layout(appearance_layout)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "turbo", "cividis", "coolwarm", "gray", "jet"])
        self.reverse_colormap = QCheckBox()
        self.scalar_auto = QCheckBox()
        self.scalar_auto.setChecked(True)
        self.scalar_min = _double_box(0.0)
        self.scalar_max = _double_box(1.0)
        self.legend_visible = QCheckBox()
        self.legend_visible.setChecked(True)
        self.legend_position = QComboBox()
        self.legend_position.addItems(["Left", "Right"])
        self.surface_opacity = _double_box(0.58, minimum=0.0, maximum=1.0, step=0.05)
        self.point_size = QSpinBox()
        self.point_size.setRange(2, 30)
        self.point_size.setValue(10)
        self.selected_color_combo = QComboBox()
        for name, value in color_choices().items():
            self.selected_color_combo.addItem(name, value)
        appearance_layout.addRow("Colormap", self.colormap_combo)
        appearance_layout.addRow("Reverse colormap", self.reverse_colormap)
        appearance_layout.addRow("Auto scalar range", self.scalar_auto)
        appearance_layout.addRow("Scalar min", self.scalar_min)
        appearance_layout.addRow("Scalar max", self.scalar_max)
        appearance_layout.addRow("Legend", self.legend_visible)
        appearance_layout.addRow("Legend position", self.legend_position)
        appearance_layout.addRow("Surface opacity", self.surface_opacity)
        appearance_layout.addRow("Point size", self.point_size)
        appearance_layout.addRow("Selected color", self.selected_color_combo)
        layout.addWidget(self.appearance_section)

        self.selection_section = _CollapsibleSection("Selection")
        selection_layout = QVBoxLayout()
        self.selection_section.set_content_layout(selection_layout)
        self.selection_summary = QLabel("Selected: 0 | Hidden: 0")
        selection_layout.addWidget(self.selection_summary)
        self.selection_table = QTableWidget(0, 8)
        self.selection_table.setHorizontalHeaderLabels(["Node", "X", "Y", "Z", "View X", "View Y", "View Z", "Value"])
        self.selection_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.selection_table.verticalHeader().setVisible(False)
        self.selection_table.setMinimumHeight(110)
        selection_layout.addWidget(self.selection_table)
        selection_buttons_1 = QHBoxLayout()
        self.clear_selection_button = QPushButton("Clear")
        self.invert_selection_button = QPushButton("Invert")
        self.hide_selected_button = QPushButton("Hide Selected")
        selection_buttons_1.addWidget(self.clear_selection_button)
        selection_buttons_1.addWidget(self.invert_selection_button)
        selection_buttons_1.addWidget(self.hide_selected_button)
        selection_layout.addLayout(selection_buttons_1)
        selection_buttons_2 = QHBoxLayout()
        self.isolate_selected = QCheckBox("Isolate selected")
        self.show_all_hidden_button = QPushButton("Show All")
        self.export_selected_button = QPushButton("Export Selected CSV")
        selection_buttons_2.addWidget(self.isolate_selected)
        selection_buttons_2.addWidget(self.show_all_hidden_button)
        selection_buttons_2.addWidget(self.export_selected_button)
        selection_layout.addLayout(selection_buttons_2)
        layout.addWidget(self.selection_section)

        self.mode_section = _CollapsibleSection("Modes")
        mode_layout = QVBoxLayout()
        self.mode_section.set_content_layout(mode_layout)
        self.mode_table = QTableWidget(0, 5)
        self.mode_table.setHorizontalHeaderLabels(["Mode", "Freq Hz", "Visc", "Hyst", "DOF"])
        self.mode_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mode_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.mode_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mode_table.verticalHeader().setVisible(False)
        self.mode_table.setMinimumHeight(150)
        mode_layout.addWidget(self.mode_table)
        mode_controls = QFormLayout()
        self.component_combo = QComboBox()
        self.component_combo.addItems(["Magnitude", "X", "Y", "Z", "Rx", "Ry", "Rz"])
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(list(ModeNormalization.LABELS.values()))
        self.deformation_scale = _double_box(1.0, minimum=-1.0e9, maximum=1.0e9, step=0.1)
        self.animate_mode = QCheckBox()
        animation_preferences = self.settings.load_animation_preferences()
        self.animation_time = _double_box(
            float(animation_preferences["duration_seconds"]),
            minimum=0.1,
            maximum=3600.0,
            step=0.1,
        )
        self.animation_time.setSuffix(" s")
        self.animation_fps = QSpinBox()
        self.animation_fps.setRange(1, 240)
        self.animation_fps.setValue(int(animation_preferences["fps"]))
        self.animation_fps.setSuffix(" fps")
        self.animation_time_label = QLabel("Animation time")
        self.animation_fps_label = QLabel("FPS")
        mode_controls.addRow("Color by", self.component_combo)
        mode_controls.addRow("Normalization", self.normalization_combo)
        mode_controls.addRow("Deformation scale", self.deformation_scale)
        mode_controls.addRow("Animate", self.animate_mode)
        mode_controls.addRow(self.animation_time_label, self.animation_time)
        mode_controls.addRow(self.animation_fps_label, self.animation_fps)
        self._set_animation_controls_visible(False)
        mode_layout.addLayout(mode_controls)
        layout.addWidget(self.mode_section)

        self.overlay_section = _CollapsibleSection("Test / FE Overlay", expanded=False)
        overlay_layout = QVBoxLayout()
        self.overlay_section.set_content_layout(overlay_layout)
        self.overlay_path_label = QLabel("No overlay loaded")
        self.overlay_path_label.setWordWrap(True)
        overlay_layout.addWidget(self.overlay_path_label)
        overlay_buttons = QHBoxLayout()
        self.load_overlay_button = QPushButton("Load Overlay")
        self.clear_overlay_button = QPushButton("Clear")
        self.match_overlay_transform_button = QPushButton("Match Primary Transform")
        overlay_buttons.addWidget(self.load_overlay_button)
        overlay_buttons.addWidget(self.clear_overlay_button)
        overlay_buttons.addWidget(self.match_overlay_transform_button)
        overlay_layout.addLayout(overlay_buttons)
        overlay_form = QFormLayout()
        self.overlay_visible = QCheckBox()
        self.overlay_visible.setChecked(True)
        self.overlay_show_points = QCheckBox()
        self.overlay_show_points.setChecked(True)
        self.overlay_show_surface = QCheckBox()
        self.overlay_show_surface.setChecked(True)
        self.overlay_opacity = _double_box(0.32, minimum=0.0, maximum=1.0, step=0.05)
        self.overlay_color_combo = QComboBox()
        for name, value in color_choices().items():
            self.overlay_color_combo.addItem(name, value)
        self.overlay_mode_combo = QComboBox()
        self.overlay_deformation_scale = _double_box(1.0, minimum=-1.0e9, maximum=1.0e9, step=0.1)
        overlay_form.addRow("Visible", self.overlay_visible)
        overlay_form.addRow("Points", self.overlay_show_points)
        overlay_form.addRow("Topology", self.overlay_show_surface)
        overlay_form.addRow("Opacity", self.overlay_opacity)
        overlay_form.addRow("Color", self.overlay_color_combo)
        overlay_form.addRow("Mode", self.overlay_mode_combo)
        overlay_form.addRow("Deformation scale", self.overlay_deformation_scale)
        overlay_layout.addLayout(overlay_form)
        overlay_transform = QFormLayout()
        self.overlay_scale_x = _double_box(1.0, minimum=-1.0e9, maximum=1.0e9, step=0.1)
        self.overlay_scale_y = _double_box(1.0, minimum=-1.0e9, maximum=1.0e9, step=0.1)
        self.overlay_scale_z = _double_box(1.0, minimum=-1.0e9, maximum=1.0e9, step=0.1)
        self.overlay_trans_x = _double_box(0.0)
        self.overlay_trans_y = _double_box(0.0)
        self.overlay_trans_z = _double_box(0.0)
        self.overlay_rot_x = _angle_box(0.0)
        self.overlay_rot_y = _angle_box(0.0)
        self.overlay_rot_z = _angle_box(0.0)
        overlay_transform.addRow("Scale X", self.overlay_scale_x)
        overlay_transform.addRow("Scale Y", self.overlay_scale_y)
        overlay_transform.addRow("Scale Z", self.overlay_scale_z)
        overlay_transform.addRow("Translate X", self.overlay_trans_x)
        overlay_transform.addRow("Translate Y", self.overlay_trans_y)
        overlay_transform.addRow("Translate Z", self.overlay_trans_z)
        overlay_transform.addRow("Rotate X", self.overlay_rot_x)
        overlay_transform.addRow("Rotate Y", self.overlay_rot_y)
        overlay_transform.addRow("Rotate Z", self.overlay_rot_z)
        overlay_layout.addLayout(overlay_transform)
        layout.addWidget(self.overlay_section)

        self.mac_section = _CollapsibleSection("MAC / Cross-MAC", expanded=False)
        mac_layout = QVBoxLayout()
        self.mac_section.set_content_layout(mac_layout)
        mac_form = QFormLayout()
        self.mac_target_combo = QComboBox()
        self.mac_target_combo.addItems(["Current model", "Overlay model"])
        self.mac_component_combo = QComboBox()
        self.mac_component_combo.addItems(["XYZ", "X", "Y", "Z", "RxRyRz", "All available"])
        self.mac_nearest_check = QCheckBox()
        self.mac_tolerance = _double_box(1.0e-6, minimum=0.0, maximum=1.0e9, step=1.0e-4)
        mac_form.addRow("Compare", self.mac_target_combo)
        mac_form.addRow("Components", self.mac_component_combo)
        mac_form.addRow("Nearest fallback", self.mac_nearest_check)
        mac_form.addRow("Tolerance", self.mac_tolerance)
        mac_layout.addLayout(mac_form)
        mac_buttons = QHBoxLayout()
        self.compute_mac_button = QPushButton("Compute MAC")
        self.export_mac_button = QPushButton("Export MAC CSV")
        mac_buttons.addWidget(self.compute_mac_button)
        mac_buttons.addWidget(self.export_mac_button)
        mac_layout.addLayout(mac_buttons)
        self.mac_table = QTableWidget(0, 0)
        self.mac_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mac_table.setMinimumHeight(150)
        mac_layout.addWidget(self.mac_table)
        layout.addWidget(self.mac_section)

        self.exports_section = _CollapsibleSection("Exports", expanded=False)
        exports_layout = QVBoxLayout()
        self.exports_section.set_content_layout(exports_layout)
        self.export_nodes_button = QPushButton("Export All Nodes CSV")
        self.export_modes_button = QPushButton("Export Mode Shapes CSV")
        self.export_screenshot_button = QPushButton("Export Screenshot PNG")
        self.export_scene_button = QPushButton("Export Scene VTK/VTU")
        self.export_animation_button = QPushButton("Export Animation MP4/AVI/GIF")
        self.export_animation_button.setEnabled(False)
        self.export_animation_button.setToolTip("Enable Animate in the Modes section to export an animation.")
        exports_layout.addWidget(self.export_nodes_button)
        exports_layout.addWidget(self.export_modes_button)
        exports_layout.addWidget(self.export_screenshot_button)
        exports_layout.addWidget(self.export_scene_button)
        exports_layout.addWidget(self.export_animation_button)
        layout.addWidget(self.exports_section)

        self.diagnostics_section = _CollapsibleSection("Diagnostics", expanded=False)
        diagnostics_layout = QVBoxLayout()
        self.diagnostics_section.set_content_layout(diagnostics_layout)
        self.diagnostics_text = QPlainTextEdit()
        self.diagnostics_text.setReadOnly(True)
        self.diagnostics_text.setMinimumHeight(120)
        diagnostics_layout.addWidget(self.diagnostics_text)
        self.copy_diagnostics_button = QPushButton("Copy Diagnostics")
        diagnostics_layout.addWidget(self.copy_diagnostics_button)
        layout.addWidget(self.diagnostics_section)

        self._apply_render_options_to_controls()
        self._apply_persisted_view_controls()
        self._update_recent_files_ui()
        self._update_selection_panel()
        self._update_diagnostics_panel()

        layout.addStretch(1)
        return container

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(self._choose_file)
        self.export_button.clicked.connect(self._export_file)
        self.open_recent_button.clicked.connect(self._open_recent_file)
        self.load_sample_button.clicked.connect(self._load_selected_sample)
        self.cs_combo.currentIndexChanged.connect(self._coordinate_system_selected)
        self.mode_table.itemSelectionChanged.connect(lambda: self.refresh_scene(reset_camera=False))
        self.component_combo.currentIndexChanged.connect(lambda: self.refresh_scene(reset_camera=False))
        self.normalization_combo.currentIndexChanged.connect(lambda: self.refresh_scene(reset_camera=False))
        self.animate_mode.toggled.connect(self._animation_toggled)
        self.clear_selection_button.clicked.connect(self._clear_selection)
        self.invert_selection_button.clicked.connect(self._invert_selection)
        self.hide_selected_button.clicked.connect(self._hide_selected)
        self.isolate_selected.toggled.connect(self._isolate_selected_toggled)
        self.show_all_hidden_button.clicked.connect(self._show_all_hidden)
        self.export_selected_button.clicked.connect(self._export_selected_nodes)
        self.load_overlay_button.clicked.connect(self._choose_overlay_file)
        self.clear_overlay_button.clicked.connect(self._clear_overlay)
        self.match_overlay_transform_button.clicked.connect(self._match_overlay_transform)
        self.overlay_mode_combo.currentIndexChanged.connect(self._overlay_mode_changed)
        self.compute_mac_button.clicked.connect(self._compute_mac)
        self.export_mac_button.clicked.connect(self._export_mac)
        self.export_nodes_button.clicked.connect(self._export_all_nodes)
        self.export_modes_button.clicked.connect(self._export_modes)
        self.export_screenshot_button.clicked.connect(self._export_screenshot)
        self.export_scene_button.clicked.connect(self._export_scene)
        self.export_animation_button.clicked.connect(self._export_animation)
        self.copy_diagnostics_button.clicked.connect(self._copy_diagnostics)

        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(self._animation_interval_ms())
        self.animation_timer.timeout.connect(self._animation_tick)
        self.animation_time.valueChanged.connect(lambda *_: self._animation_preferences_changed())
        self.animation_fps.valueChanged.connect(lambda *_: self._animation_preferences_changed())

        for widget in [
            self.scale_x,
            self.scale_y,
            self.scale_z,
            self.trans_x,
            self.trans_y,
            self.trans_z,
            self.origin_x,
            self.origin_y,
            self.origin_z,
            self.point_size,
            self.deformation_scale,
        ]:
            widget.valueChanged.connect(lambda *_: self.refresh_scene(reset_camera=False))
        for box in [self.rot_x, self.rot_y, self.rot_z]:
            box.valueChanged.connect(lambda *_: self.refresh_scene(reset_camera=False))
        for box in [self.show_points, self.show_surface, self.generate_surface, self.show_traces]:
            box.toggled.connect(lambda *_: self.refresh_scene(reset_camera=False))
        for widget in [
            self.colormap_combo,
            self.reverse_colormap,
            self.scalar_auto,
            self.scalar_min,
            self.scalar_max,
            self.legend_visible,
            self.legend_position,
            self.surface_opacity,
            self.point_size,
            self.selected_color_combo,
        ]:
            signal = widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged if isinstance(widget, (QDoubleSpinBox, QSpinBox)) else widget.toggled
            signal.connect(lambda *_: self._appearance_changed())
        for widget in [
            self.overlay_visible,
            self.overlay_show_points,
            self.overlay_show_surface,
            self.overlay_opacity,
            self.overlay_color_combo,
            self.overlay_deformation_scale,
            self.overlay_scale_x,
            self.overlay_scale_y,
            self.overlay_scale_z,
            self.overlay_trans_x,
            self.overlay_trans_y,
            self.overlay_trans_z,
            self.overlay_rot_x,
            self.overlay_rot_y,
            self.overlay_rot_z,
        ]:
            signal = widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged if isinstance(widget, (QDoubleSpinBox, QSpinBox)) else widget.toggled
            signal.connect(lambda *_: self._overlay_controls_changed())

    def load_file(self, path: Path) -> None:
        try:
            self.model = load_unv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.current_path = path
        self.path_label.setText(str(path))
        self.export_button.setEnabled(True)
        self._update_animation_export_state()
        self.settings.add_recent_file(path)
        self.settings.set_last_folder(path)
        self.selection = SelectionState()
        self.isolate_selected.blockSignals(True)
        self.isolate_selected.setChecked(False)
        self.isolate_selected.blockSignals(False)
        self._update_recent_files_ui()
        self._populate_coordinate_systems()
        self._populate_modes()
        self._update_summary()
        self._record_message("INFO", f"Loaded {path.name}")
        self.statusBar().showMessage(f"Loaded {path.name}")
        self.refresh_scene(reset_camera=True)

    def refresh_scene(self, reset_camera: bool = False) -> None:
        self._scene_meshes.clear()
        self.plotter.clear()
        self.plotter.set_background("#15181d")
        self.plotter.add_axes()

        if self.model is None or not self.model.nodes:
            self.plotter.add_text(
                "Open a UNV/UFF modal-test file",
                position="upper_left",
                font_size=12,
                color="white",
                name="hover_text",
            )
            self.plotter.render()
            return

        phase = self._phase if self.animate_mode.isChecked() else 1.0
        frame = self._compute_primary_frame(phase)
        if frame is None:
            return
        labels, all_points, all_scalars, visible_labels, points, scalars = frame
        model = self.model
        assert model is not None
        self._all_current_points = all_points
        self._all_current_scalars = all_scalars
        self._all_current_labels = labels
        view_model = _view_model(model, visible_labels)

        self._current_points = points
        self._current_scalars = scalars
        self._current_labels = visible_labels

        mode = self.selected_mode()
        scalar_title = self.component_combo.currentText() if mode is not None else "Node"
        options = self.current_render_options()
        scalar_clim = self._effective_scalar_clim(options, scalar_title)
        surface = None
        if self.show_surface.isChecked():
            surface = element_surface(view_model, points)
        if surface is None and self.generate_surface.isChecked():
            surface = generated_surface(view_model, points)
        if surface is not None:
            surface.point_data["value"] = scalars
            self._scene_meshes["surface"] = surface
            self.plotter.add_mesh(
                surface,
                scalars="value",
                cmap=options.pyvista_colormap,
                clim=scalar_clim,
                opacity=options.surface_opacity,
                smooth_shading=True,
                scalar_bar_args=_scalar_bar_args(scalar_title, options.legend_position),
                show_scalar_bar=options.legend_visible,
            )

        if self.show_traces.isChecked():
            traces = trace_line_mesh(view_model, points)
            if traces is not None:
                self._scene_meshes["traces"] = traces
                self.plotter.add_mesh(traces, color="#d8dee9", line_width=3, render_lines_as_tubes=True)

        if self.show_points.isChecked():
            cloud = point_cloud(view_model, points)
            cloud.point_data["value"] = scalars
            self._scene_meshes["points"] = cloud
            self.plotter.add_mesh(
                cloud,
                scalars="value",
                cmap=options.pyvista_colormap,
                clim=scalar_clim,
                point_size=options.point_size,
                render_points_as_spheres=True,
                scalar_bar_args=_scalar_bar_args(scalar_title, options.legend_position),
                show_scalar_bar=options.legend_visible and surface is None,
            )

        self._render_selected_points(view_model, points, visible_labels, options)
        self._render_overlay(phase)
        self.plotter.add_text(
            "Hover over a point",
            position="upper_left",
            font_size=10,
            color="white",
            name="hover_text",
        )
        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()
        self._update_selection_panel()

    def current_transform(self) -> TransformSpec:
        rotation = rotation_matrix_from_euler_degrees(self.rot_x.value(), self.rot_y.value(), self.rot_z.value())
        return TransformSpec(
            scale=np.array([self.scale_x.value(), self.scale_y.value(), self.scale_z.value()]),
            translation=np.array([self.trans_x.value(), self.trans_y.value(), self.trans_z.value()]),
            cs_rotation=rotation,
            cs_origin=np.array([self.origin_x.value(), self.origin_y.value(), self.origin_z.value()]),
        )

    def current_mode_normalization(self) -> str:
        return ModeNormalization.BY_LABEL.get(self.normalization_combo.currentText(), ModeNormalization.RAW)

    def current_render_options(self) -> RenderOptions:
        return RenderOptions(
            colormap=self.colormap_combo.currentText(),
            reverse_colormap=self.reverse_colormap.isChecked(),
            scalar_auto=self.scalar_auto.isChecked(),
            scalar_min=self.scalar_min.value(),
            scalar_max=self.scalar_max.value(),
            legend_visible=self.legend_visible.isChecked(),
            legend_position=self.legend_position.currentText(),
            surface_opacity=self.surface_opacity.value(),
            point_size=self.point_size.value(),
            selected_color=str(self.selected_color_combo.currentData()),
        )

    def _effective_scalar_clim(self, options: RenderOptions, scalar_title: str) -> tuple[float, float] | None:
        if options.clim is not None:
            return options.clim
        if (
            self.current_mode_normalization() == ModeNormalization.MAX_DISPLACEMENT
            and scalar_title == "Magnitude"
        ):
            return (0.0, 1.0)
        return None

    def _compute_primary_frame(
        self,
        phase: float,
    ) -> tuple[list[int], np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray] | None:
        if self.model is None or not self.model.nodes:
            return None

        model = self.model
        transformed = transformed_node_coordinates(model, self.current_transform())
        labels = model.node_labels
        base_points = np.vstack([transformed[label] for label in labels])
        mode = self.selected_mode()
        all_points, all_scalars = deformed_points(
            model,
            mode,
            self.deformation_scale.value() * phase,
            self.component_combo.currentText(),
            base_points=base_points,
            normalization=self.current_mode_normalization(),
        )
        if mode is None:
            all_scalars = np.array(labels, dtype=float)

        visible_labels = self.selection.visible_labels(labels)
        label_index = {label: index for index, label in enumerate(labels)}
        visible_indices = [label_index[label] for label in visible_labels]
        points = all_points[visible_indices] if visible_indices else np.empty((0, 3), dtype=float)
        scalars = all_scalars[visible_indices] if visible_indices else np.empty(0, dtype=float)
        return labels, all_points, all_scalars, visible_labels, points, scalars

    def current_overlay_transform(self) -> TransformSpec:
        rotation = rotation_matrix_from_euler_degrees(
            self.overlay_rot_x.value(),
            self.overlay_rot_y.value(),
            self.overlay_rot_z.value(),
        )
        return TransformSpec(
            scale=np.array([self.overlay_scale_x.value(), self.overlay_scale_y.value(), self.overlay_scale_z.value()]),
            translation=np.array([self.overlay_trans_x.value(), self.overlay_trans_y.value(), self.overlay_trans_z.value()]),
            cs_rotation=rotation,
        )

    def _compute_overlay_frame(self, phase: float) -> tuple[ModalModel, np.ndarray] | None:
        if self.overlay.model is None or not self.overlay.visible:
            return None
        model = self.overlay.model
        transformed = transformed_node_coordinates(model, self.current_overlay_transform())
        labels = model.node_labels
        if not labels:
            return None
        base_points = np.vstack([transformed[label] for label in labels])
        points, _ = deformed_points(
            model,
            self.overlay.selected_mode(),
            self.overlay.deformation_scale * phase,
            "Magnitude",
            base_points=base_points,
            normalization=self.current_mode_normalization(),
        )
        return model, points

    def selected_mode(self) -> ModeShape | None:
        if self.model is None:
            return None
        row = self.mode_table.currentRow()
        if row < 0 or row >= len(self.model.modes):
            return None
        return self.model.modes[row]

    def _populate_coordinate_systems(self) -> None:
        self.cs_combo.blockSignals(True)
        self.cs_combo.clear()
        self.cs_combo.addItem("Global / Manual", None)
        if self.model is not None:
            for label, cs in sorted(self.model.coordinate_systems.items()):
                self.cs_combo.addItem(f"{label}: {cs.name or 'Coordinate system'}", label)
        self.cs_combo.blockSignals(False)
        self._set_coordinate_system(None)

    def _populate_modes(self) -> None:
        self.mode_table.setRowCount(0)
        if self.model is None:
            return
        self.mode_table.setRowCount(len(self.model.modes))
        for row, mode in enumerate(self.model.modes):
            values = [
                "" if mode.mode_number is None else str(mode.mode_number),
                _fmt(mode.frequency_hz),
                _fmt(mode.viscous_damping),
                _fmt(mode.hysteretic_damping),
                ",".join(mode.component_names),
            ]
            for column, value in enumerate(values):
                self.mode_table.setItem(row, column, QTableWidgetItem(value))
        if self.model.modes:
            self.mode_table.selectRow(0)
        self.mode_table.resizeColumnsToContents()

    def _update_summary(self) -> None:
        if self.model is None:
            self.summary_label.setText("Datasets: none")
            return
        counts = self.model.metadata.get("dataset_counts", {})
        dataset_summary = ", ".join(f"{key}: {counts[key]}" for key in sorted(counts))
        surface_count = supported_surface_elements(self.model.elements)
        text = (
            f"Nodes: {len(self.model.nodes)} | Modes: {len(self.model.modes)} | "
            f"FRFs: {len(self.model.functions)} | Surfaces: {surface_count}\n"
            f"Datasets: {dataset_summary or 'none'}"
        )
        if self.model.units is not None:
            text += f"\nUnits: {self.model.units.description or self.model.units.code}"
        if self.model.diagnostics:
            text += f"\nDiagnostics: {len(self.model.diagnostics)} preserved/read notes"
        self.summary_label.setText(text)

    def _choose_file(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open UNV/UFF modal-test file",
            str(self.current_path.parent if self.current_path else self.settings.last_folder()),
            "UNV/UFF files (*.unv *.uff *.unv.txt *.uff.txt);;All files (*.*)",
        )
        if file_name:
            self.load_file(Path(file_name))

    def _export_file(self) -> None:
        if self.model is None:
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export modified UNV/UFF",
            str((self.current_path or Path("modified.unv")).with_name("modified.unv")),
            "UNV/UFF files (*.unv *.uff);;All files (*.*)",
        )
        if not file_name:
            return
        transform_vectors = ExportDialog.ask(self)
        if transform_vectors is None:
            return
        try:
            export_unv(self.model, file_name, self.current_transform(), transform_vectors=transform_vectors)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {Path(file_name).name}")
        self._record_message("INFO", f"Exported modified UNV/UFF to {file_name}")

    def _open_recent_file(self) -> None:
        path = self.recent_combo.currentData()
        if path:
            self.load_file(Path(path))

    def _load_selected_sample(self) -> None:
        name = self.sample_combo.currentText()
        try:
            path = ensure_sample_file(name)
        except Exception as exc:
            QMessageBox.warning(self, "Sample unavailable", f"Could not load sample {name}:\n{exc}")
            self._record_message("WARN", f"Sample {name} unavailable: {exc}")
            return
        self.load_file(path)

    def _choose_overlay_file(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open overlay UNV/UFF model",
            str(self.current_path.parent if self.current_path else self.settings.last_folder()),
            "UNV/UFF files (*.unv *.uff *.unv.txt *.uff.txt);;All files (*.*)",
        )
        if file_name:
            self._load_overlay_file(Path(file_name))

    def _load_overlay_file(self, path: Path) -> None:
        try:
            model = load_unv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Overlay load failed", str(exc))
            return
        self.overlay.model = model
        self.overlay.path = path
        self.overlay_path_label.setText(str(path))
        self._populate_overlay_modes()
        self._record_message("INFO", f"Loaded overlay {path.name}")
        self.refresh_scene(reset_camera=False)

    def _clear_overlay(self) -> None:
        self.overlay = OverlayState()
        self.overlay_path_label.setText("No overlay loaded")
        self.overlay_mode_combo.clear()
        self.refresh_scene(reset_camera=False)

    def _match_overlay_transform(self) -> None:
        for source, target in [
            (self.scale_x, self.overlay_scale_x),
            (self.scale_y, self.overlay_scale_y),
            (self.scale_z, self.overlay_scale_z),
            (self.trans_x, self.overlay_trans_x),
            (self.trans_y, self.overlay_trans_y),
            (self.trans_z, self.overlay_trans_z),
            (self.rot_x, self.overlay_rot_x),
            (self.rot_y, self.overlay_rot_y),
            (self.rot_z, self.overlay_rot_z),
        ]:
            target.setValue(source.value())
        self._overlay_controls_changed()

    def _populate_overlay_modes(self) -> None:
        self.overlay_mode_combo.blockSignals(True)
        self.overlay_mode_combo.clear()
        self.overlay_mode_combo.addItem("Undeformed", -1)
        if self.overlay.model is not None:
            for index, mode in enumerate(self.overlay.model.modes):
                self.overlay_mode_combo.addItem(mode_label(mode, index), index)
        self.overlay_mode_combo.blockSignals(False)
        self.overlay.mode_index = -1

    def _overlay_mode_changed(self) -> None:
        data = self.overlay_mode_combo.currentData()
        self.overlay.mode_index = -1 if data is None else int(data)
        self.refresh_scene(reset_camera=False)

    def _overlay_controls_changed(self) -> None:
        self.overlay.visible = self.overlay_visible.isChecked()
        self.overlay.show_points = self.overlay_show_points.isChecked()
        self.overlay.show_surface = self.overlay_show_surface.isChecked()
        self.overlay.opacity = self.overlay_opacity.value()
        self.overlay.color = str(self.overlay_color_combo.currentData())
        self.overlay.deformation_scale = self.overlay_deformation_scale.value()
        self.overlay.transform = self.current_overlay_transform()
        self.refresh_scene(reset_camera=False)

    def _appearance_changed(self) -> None:
        self.render_options = self.current_render_options()
        self.refresh_scene(reset_camera=False)

    def _clear_selection(self) -> None:
        self.selection.clear_selection()
        self.refresh_scene(reset_camera=False)

    def _invert_selection(self) -> None:
        if self.model is not None:
            self.selection.invert(self.model.node_labels)
        self.refresh_scene(reset_camera=False)

    def _hide_selected(self) -> None:
        self.selection.hide_selected()
        self.refresh_scene(reset_camera=False)

    def _isolate_selected_toggled(self, checked: bool) -> None:
        self.selection.isolate_selected = checked
        self.refresh_scene(reset_camera=False)

    def _show_all_hidden(self) -> None:
        self.selection.show_all()
        self.isolate_selected.blockSignals(True)
        self.isolate_selected.setChecked(False)
        self.isolate_selected.blockSignals(False)
        self.refresh_scene(reset_camera=False)

    def _export_all_nodes(self) -> None:
        if self.model is None:
            return
        file_name = self._choose_save_file("Export all nodes CSV", "nodes.csv", "CSV files (*.csv)")
        if file_name:
            export_nodes_csv(self.model, file_name, self.current_transform(), scalars=self._scalar_map())
            self._record_message("INFO", f"Exported node CSV to {file_name}")

    def _export_selected_nodes(self) -> None:
        if self.model is None or not self.selection.selected_node_ids:
            return
        file_name = self._choose_save_file("Export selected nodes CSV", "selected_nodes.csv", "CSV files (*.csv)")
        if file_name:
            labels = sorted(self.selection.selected_node_ids)
            export_nodes_csv(self.model, file_name, self.current_transform(), labels, self._scalar_map())
            self._record_message("INFO", f"Exported selected-node CSV to {file_name}")

    def _export_modes(self) -> None:
        if self.model is None:
            return
        file_name = self._choose_save_file("Export mode shapes CSV", "mode_shapes.csv", "CSV files (*.csv)")
        if file_name:
            export_modes_csv(self.model, file_name)
            self._record_message("INFO", f"Exported mode-shape CSV to {file_name}")

    def _export_mac(self) -> None:
        if self._mac_matrix.size == 0:
            self._compute_mac()
        if self._mac_matrix.size == 0:
            return
        file_name = self._choose_save_file("Export MAC CSV", "mac.csv", "CSV files (*.csv)")
        if file_name:
            export_mac_csv(file_name, self._mac_matrix, self._mac_row_modes, self._mac_column_modes)
            self._record_message("INFO", f"Exported MAC CSV to {file_name}")

    def _export_screenshot(self) -> None:
        file_name = self._choose_save_file("Export screenshot PNG", "scene.png", "PNG files (*.png)")
        if file_name:
            try:
                export_screenshot(self.plotter, file_name)
                self._record_message("INFO", f"Exported screenshot to {file_name}")
            except Exception as exc:
                QMessageBox.warning(self, "Screenshot failed", str(exc))
                self._record_message("WARN", f"Screenshot export failed: {exc}")

    def _export_scene(self) -> None:
        if self.model is None or self._all_current_points.size == 0:
            return
        file_name = self._choose_save_file("Export scene VTK/VTU", "scene.vtp", "VTK files (*.vtp *.vtk *.vtu)")
        if file_name:
            export_scene_vtk(
                self.model,
                file_name,
                self._all_current_points,
                self._all_current_scalars,
                self.selection.selected_node_ids,
                self.selection.hidden_node_ids,
            )
            self._record_message("INFO", f"Exported scene geometry to {file_name}")

    def _export_animation(self) -> None:
        if self.model is None or not self.animate_mode.isChecked():
            QMessageBox.information(self, "Animation export unavailable", "Enable Animate before exporting media.")
            return
        file_name = self._choose_save_file(
            "Export animation",
            "mode_animation.mp4",
            "Animation files (*.mp4 *.avi *.gif)",
        )
        if not file_name:
            return

        path = Path(file_name)
        if not path.suffix:
            path = path.with_suffix(".mp4")
        if path.suffix.lower() not in SUPPORTED_ANIMATION_EXTENSIONS:
            QMessageBox.warning(self, "Animation export failed", "Choose an .mp4, .avi, or .gif output file.")
            return

        was_running = self.animation_timer.isActive()
        original_phase = self._phase
        if was_running:
            self.animation_timer.stop()
        try:
            frames = export_animation_media(
                path,
                self._capture_animation_frame,
                self._animation_duration_seconds(),
                self.animation_fps.value(),
            )
            self._record_message("INFO", f"Exported animation media to {path} ({frames} frames)")
            self.statusBar().showMessage(f"Exported {path.name}")
        except Exception as exc:
            QMessageBox.warning(self, "Animation export failed", str(exc))
            self._record_message("WARN", f"Animation export failed: {exc}")
        finally:
            self._phase = original_phase
            if not self._update_animation_frame():
                self.refresh_scene(reset_camera=False)
            if was_running:
                self._animation_started_at = time.perf_counter()
                self.animation_timer.start()

    def _choose_save_file(self, title: str, default_name: str, filter_text: str) -> str:
        start = self.current_path.with_name(default_name) if self.current_path else self.settings.last_folder() / default_name
        file_name, _ = QFileDialog.getSaveFileName(self, title, str(start), f"{filter_text};;All files (*.*)")
        return file_name

    def _coordinate_system_selected(self) -> None:
        label = self.cs_combo.currentData()
        self._set_coordinate_system(None if label is None or self.model is None else self.model.coordinate_systems[label])
        self.refresh_scene(reset_camera=False)

    def _set_coordinate_system(self, cs: CoordinateSystem | None) -> None:
        rotation = np.eye(3) if cs is None else cs.rotation
        origin = np.zeros(3) if cs is None else cs.origin
        angles = euler_degrees_from_rotation_matrix(rotation)
        for box, value in zip([self.rot_x, self.rot_y, self.rot_z], angles, strict=False):
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)
        for box, value in zip([self.origin_x, self.origin_y, self.origin_z], origin, strict=False):
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)

    def _compute_mac(self) -> None:
        if self.model is None or not self.model.modes:
            return
        target_overlay = self.mac_target_combo.currentText() == "Overlay model"
        if target_overlay:
            if self.overlay.model is None or not self.overlay.model.modes:
                QMessageBox.information(self, "MAC unavailable", "Load an overlay model with modes first.")
                return
            comparison = self.overlay.model
            column_modes = comparison.modes
        else:
            comparison = self.model
            column_modes = self.model.modes

        options = MacOptions(
            components=self.mac_component_combo.currentText(),
            use_nearest_fallback=self.mac_nearest_check.isChecked(),
            nearest_tolerance=self.mac_tolerance.value(),
        )
        pairs = pair_nodes_for_mac(self.model, comparison, options)
        if not pairs:
            QMessageBox.information(self, "MAC unavailable", "No paired nodes were found for MAC calculation.")
            return

        self._mac_row_modes = self.model.modes
        self._mac_column_modes = column_modes
        self._mac_matrix = compute_mac_matrix(self._mac_row_modes, self._mac_column_modes, pairs, options.components)
        self._populate_mac_table()
        self._record_message("INFO", f"Computed MAC using {len(pairs)} paired nodes and {options.components} components")

    def _populate_mac_table(self) -> None:
        matrix = self._mac_matrix
        self.mac_table.setRowCount(matrix.shape[0])
        self.mac_table.setColumnCount(matrix.shape[1])
        self.mac_table.setVerticalHeaderLabels(
            [mode_label(mode, row) for row, mode in enumerate(self._mac_row_modes)]
        )
        self.mac_table.setHorizontalHeaderLabels(
            [mode_label(mode, column) for column, mode in enumerate(self._mac_column_modes)]
        )
        best = {row: column for row, column, _ in best_mac_matches(matrix)}
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                item = QTableWidgetItem(f"{matrix[row, column]:.4f}")
                if best.get(row) == column:
                    item.setBackground(QColor("#315c3b"))
                self.mac_table.setItem(row, column, item)
        self.mac_table.resizeColumnsToContents()

    def _render_selected_points(
        self,
        view_model: ModalModel,
        points: np.ndarray,
        visible_labels: list[int],
        options: RenderOptions,
    ) -> None:
        selected_indices = [
            index for index, label in enumerate(visible_labels) if label in self.selection.selected_node_ids
        ]
        if not selected_indices:
            return
        selected_points = points[selected_indices]
        selected_model = _view_model(view_model, [visible_labels[index] for index in selected_indices])
        mesh = point_cloud(selected_model, selected_points)
        self._scene_meshes["selected_points"] = mesh
        self.plotter.add_mesh(
            mesh,
            color=options.selected_color,
            point_size=options.point_size + 6,
            render_points_as_spheres=True,
        )

    def _render_overlay(self, phase: float) -> None:
        frame = self._compute_overlay_frame(phase)
        if frame is None:
            return
        model, points = frame
        if self.overlay.show_surface:
            surface = element_surface(model, points)
            if surface is None:
                surface = generated_surface(model, points)
            if surface is not None:
                self._scene_meshes["overlay_surface"] = surface
                self.plotter.add_mesh(
                    surface,
                    color=self.overlay.color,
                    opacity=self.overlay.opacity,
                    smooth_shading=True,
                    show_scalar_bar=False,
                )
        if self.overlay.show_points:
            cloud = point_cloud(model, points)
            self._scene_meshes["overlay_points"] = cloud
            self.plotter.add_mesh(
                cloud,
                color=self.overlay.color,
                opacity=min(1.0, self.overlay.opacity + 0.2),
                point_size=max(4, self.point_size.value() - 1),
                render_points_as_spheres=True,
            )

    def _select_point_index(self, point_id: int, toggle: bool = False) -> None:
        if point_id < 0 or point_id >= len(self._current_labels):
            return
        label = self._current_labels[point_id]
        if toggle:
            self.selection.toggle(label)
        else:
            self.selection.select_only(label)
        self.refresh_scene(reset_camera=False)

    def _update_selection_panel(self) -> None:
        self.selection_summary.setText(
            f"Selected: {len(self.selection.selected_node_ids)} | Hidden: {len(self.selection.hidden_node_ids)}"
        )
        self.selection_table.setRowCount(0)
        if self.model is None:
            return
        transformed = transformed_node_coordinates(self.model, self.current_transform())
        scalars = self._scalar_map()
        labels = [label for label in sorted(self.selection.selected_node_ids) if label in self.model.nodes]
        self.selection_table.setRowCount(len(labels))
        for row, label in enumerate(labels):
            original = self.model.nodes[label].coordinates
            current = transformed.get(label, original)
            values = [
                str(label),
                *[_fmt(value) for value in original],
                *[_fmt(value) for value in current],
                _fmt(scalars.get(label)),
            ]
            for column, value in enumerate(values):
                self.selection_table.setItem(row, column, QTableWidgetItem(value))
        self.selection_table.resizeColumnsToContents()

    def _update_recent_files_ui(self) -> None:
        self.recent_combo.blockSignals(True)
        self.recent_combo.clear()
        for path in self.settings.recent_files():
            self.recent_combo.addItem(path.name, str(path))
        self.open_recent_button.setEnabled(self.recent_combo.count() > 0)
        self.recent_combo.blockSignals(False)

    def _apply_render_options_to_controls(self) -> None:
        options = self.render_options
        self.colormap_combo.setCurrentText(options.colormap)
        self.reverse_colormap.setChecked(options.reverse_colormap)
        self.scalar_auto.setChecked(options.scalar_auto)
        self.scalar_min.setValue(options.scalar_min)
        self.scalar_max.setValue(options.scalar_max)
        self.legend_visible.setChecked(options.legend_visible)
        self.legend_position.setCurrentText(options.legend_position)
        self.surface_opacity.setValue(options.surface_opacity)
        self.point_size.setValue(options.point_size)
        index = self.selected_color_combo.findData(options.selected_color)
        if index >= 0:
            self.selected_color_combo.setCurrentIndex(index)

    def _apply_persisted_view_controls(self) -> None:
        flags = self.settings.load_view_flags()
        self.show_points.setChecked(flags["points"])
        self.show_surface.setChecked(flags["surface"])
        self.generate_surface.setChecked(flags["generated_surface"])
        self.show_traces.setChecked(flags["traces"])
        overlay = self.settings.load_overlay_preferences()
        self.overlay_opacity.setValue(float(overlay["opacity"]))
        index = self.overlay_color_combo.findData(str(overlay["color"]))
        if index >= 0:
            self.overlay_color_combo.setCurrentIndex(index)
        self.overlay.opacity = self.overlay_opacity.value()
        self.overlay.color = str(self.overlay_color_combo.currentData())

    def _restore_persisted_ui(self) -> None:
        self.settings.restore_window(self, self.splitter, self._sections())

    def _sections(self) -> dict[str, "_CollapsibleSection"]:
        return {
            "file": self.file_section,
            "transform": self.transform_section,
            "rotation": self.rotation_section,
            "view": self.view_section,
            "appearance": self.appearance_section,
            "selection": self.selection_section,
            "modes": self.mode_section,
            "overlay": self.overlay_section,
            "mac": self.mac_section,
            "exports": self.exports_section,
            "diagnostics": self.diagnostics_section,
        }

    def _scalar_map(self) -> dict[int, float]:
        return {label: float(value) for label, value in zip(self._all_current_labels, self._all_current_scalars, strict=False)}

    def _record_message(self, level: str, message: str) -> None:
        self._messages.append(f"{level}: {message}")
        self._messages = self._messages[-200:]
        self._update_diagnostics_panel()

    def _update_diagnostics_panel(self) -> None:
        lines: list[str] = []
        if self.model is not None:
            counts = self.model.metadata.get("dataset_counts", {})
            unsupported = {key: value for key, value in counts.items() if int(key) not in SUPPORTED_DATASETS}
            lines.append(f"File: {self.current_path or ''}")
            lines.append(f"Datasets: {counts}")
            lines.append(f"Unsupported/preserved datasets: {unsupported or 'none'}")
            lines.extend(self.model.diagnostics)
        lines.extend(self._messages)
        self.diagnostics_text.setPlainText("\n".join(lines))

    def _copy_diagnostics(self) -> None:
        QApplication.clipboard().setText(self.diagnostics_text.toPlainText())

    def _animation_interval_ms(self) -> int:
        return max(1, int(round(1000.0 / max(1, self.animation_fps.value()))))

    def _animation_duration_seconds(self) -> float:
        return max(0.1, float(self.animation_time.value()))

    def _animation_preferences_changed(self) -> None:
        self.settings.save_animation_preferences(self.animation_time.value(), self.animation_fps.value())
        self.animation_timer.setInterval(self._animation_interval_ms())

    def _set_animation_controls_visible(self, visible: bool) -> None:
        for widget in [
            self.animation_time_label,
            self.animation_time,
            self.animation_fps_label,
            self.animation_fps,
        ]:
            widget.setVisible(visible)

    def _update_animation_export_state(self, animate_enabled: bool | None = None) -> None:
        animate_enabled = self.animate_mode.isChecked() if animate_enabled is None else animate_enabled
        enabled = self.model is not None and animate_enabled
        self.export_animation_button.setEnabled(enabled)

    def _animation_toggled(self, enabled: bool) -> None:
        self._set_animation_controls_visible(enabled)
        self._update_animation_export_state(enabled)
        self._phase = 0.0 if enabled else 1.0
        if enabled:
            self._animation_preferences_changed()
            self._animation_started_at = time.perf_counter()
            if not self._update_animation_frame():
                self.refresh_scene(reset_camera=False)
            self.animation_timer.start()
        else:
            self.animation_timer.stop()
            self.refresh_scene(reset_camera=False)

    def _animation_tick(self) -> None:
        elapsed = time.perf_counter() - self._animation_started_at
        self._phase = float(np.sin(2.0 * np.pi * elapsed / self._animation_duration_seconds()))
        if not self._update_animation_frame():
            self.refresh_scene(reset_camera=False)

    def _capture_animation_frame(self, phase: float) -> np.ndarray:
        self._phase = float(phase)
        if not self._update_animation_frame():
            self.refresh_scene(reset_camera=False)
        QApplication.processEvents()
        image = self.plotter.screenshot(return_img=True)
        if image is None:
            raise RuntimeError("PyVista did not return an animation frame image.")
        return np.asarray(image)

    def _update_animation_frame(self) -> bool:
        if self.model is None or not self.model.nodes:
            return False

        frame = self._compute_primary_frame(self._phase)
        if frame is None:
            return False
        labels, all_points, all_scalars, visible_labels, points, scalars = frame
        if visible_labels != self._current_labels:
            return False

        for key in ("surface", "traces", "points"):
            if key in self._scene_meshes and not self._set_mesh_points(key, points):
                return False

        if "selected_points" in self._scene_meshes:
            selected_indices = [
                index for index, label in enumerate(visible_labels) if label in self.selection.selected_node_ids
            ]
            selected_points = points[selected_indices] if selected_indices else np.empty((0, 3), dtype=float)
            if not self._set_mesh_points("selected_points", selected_points):
                return False

        overlay_frame = self._compute_overlay_frame(self._phase)
        if overlay_frame is not None:
            _, overlay_points = overlay_frame
            for key in ("overlay_surface", "overlay_points"):
                if key in self._scene_meshes and not self._set_mesh_points(key, overlay_points):
                    return False
        elif any(key.startswith("overlay_") for key in self._scene_meshes):
            return False

        self._all_current_points = all_points
        self._all_current_scalars = all_scalars
        self._all_current_labels = labels
        self._current_points = points
        self._current_scalars = scalars
        self._current_labels = visible_labels
        self.plotter.render()
        return True

    def _set_mesh_points(self, key: str, points: np.ndarray) -> bool:
        mesh = self._scene_meshes.get(key)
        if mesh is None or not hasattr(mesh, "points"):
            return False
        try:
            current = np.asarray(mesh.points)
            new_points = np.asarray(points, dtype=float)
            if current.shape != new_points.shape:
                return False
            mesh.points = new_points.copy()
        except Exception:
            return False
        return True

    def _install_hover_observer(self) -> None:
        if self._hover_observer_installed or self._point_picker is None:
            return
        try:
            self.plotter.iren.add_observer("MouseMoveEvent", self._on_mouse_move)
            self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_left_button_press)
            self._hover_observer_installed = True
        except Exception:
            try:
                self.plotter.iren.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
                self.plotter.iren.interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press)
                self._hover_observer_installed = True
            except Exception:
                self._hover_observer_installed = False

    def _on_mouse_move(self, obj: object, event: object) -> None:
        if self._point_picker is None or self._current_points.size == 0:
            return
        try:
            position = self.plotter.iren.get_event_position()
        except Exception:
            try:
                position = self.plotter.iren.interactor.GetEventPosition()
            except Exception:
                return
        try:
            self._point_picker.Pick(position[0], position[1], 0, self.plotter.renderer)
            point_id = int(self._point_picker.GetPointId())
        except Exception:
            return
        if point_id < 0 or point_id >= len(self._current_labels):
            return
        label = self._current_labels[point_id]
        x, y, z = self._current_points[point_id]
        value = self._current_scalars[point_id] if point_id < len(self._current_scalars) else float("nan")
        self.plotter.add_text(
            f"Node {label}   X {x:.6g}   Y {y:.6g}   Z {z:.6g}   Value {value:.6g}",
            position="upper_left",
            font_size=10,
            color="white",
            name="hover_text",
        )

    def _on_left_button_press(self, obj: object, event: object) -> None:
        if self._point_picker is None or self._current_points.size == 0:
            return
        try:
            position = self.plotter.iren.get_event_position()
        except Exception:
            try:
                position = self.plotter.iren.interactor.GetEventPosition()
            except Exception:
                return
        try:
            self._point_picker.Pick(position[0], position[1], 0, self.plotter.renderer)
            point_id = int(self._point_picker.GetPointId())
        except Exception:
            return
        toggle = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        self._select_point_index(point_id, toggle=toggle)

    def closeEvent(self, event: object) -> None:
        self.settings.save_render_options(self.current_render_options())
        self.settings.save_view_flags(
            self.show_points.isChecked(),
            self.show_surface.isChecked(),
            self.generate_surface.isChecked(),
            self.show_traces.isChecked(),
        )
        self.settings.save_overlay_preferences(self.overlay_opacity.value(), str(self.overlay_color_combo.currentData()))
        self.settings.save_animation_preferences(self.animation_time.value(), self.animation_fps.value())
        self.settings.save_window(self, self.splitter, self._sections())
        super().closeEvent(event)


class ExportDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        layout = QVBoxLayout(self)
        self.vector_check = QCheckBox("Transform dataset 55 / 2414 mode-shape vectors")
        self.vector_check.setChecked(False)
        layout.addWidget(self.vector_check)
        note = QLabel("Coordinate blocks are always exported with the current scale, translation, and CS alignment.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @classmethod
    def ask(cls, parent: QWidget) -> bool | None:
        dialog = cls(parent)
        if dialog.exec() != QDialog.Accepted:
            return None
        return dialog.vector_check.isChecked()


class _CollapsibleSection(QWidget):
    def __init__(self, title: str, expanded: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.header = QToolButton()
        self.header.setObjectName("SectionHeader")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.content = QFrame()
        self.content.setObjectName("SectionContent")
        self.content.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.content)

        self.header.toggled.connect(self._set_expanded)

    def set_content_layout(self, layout: QLayout) -> None:
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.content.setLayout(layout)

    def is_expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)
        self.updateGeometry()


def set_fusion_theme(app: QApplication) -> None:
    fusion = QStyleFactory.create("Fusion")
    app.setStyle(fusion if fusion is not None else "Fusion")
    app.setProperty("unv_modal_viewer_theme", "fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#20242b"))
    palette.setColor(QPalette.WindowText, QColor("#eef2f7"))
    palette.setColor(QPalette.Base, QColor("#15181d"))
    palette.setColor(QPalette.AlternateBase, QColor("#232832"))
    palette.setColor(QPalette.ToolTipBase, QColor("#eef2f7"))
    palette.setColor(QPalette.ToolTipText, QColor("#15181d"))
    palette.setColor(QPalette.Text, QColor("#eef2f7"))
    palette.setColor(QPalette.Button, QColor("#2d3440"))
    palette.setColor(QPalette.ButtonText, QColor("#eef2f7"))
    palette.setColor(QPalette.BrightText, QColor("#ff6b6b"))
    palette.setColor(QPalette.Highlight, QColor("#4c8bf5"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QWidget { font-size: 10pt; }
        QToolButton#SectionHeader {
            text-align: left;
            padding: 8px 10px;
            border: 1px solid #3a4250;
            border-radius: 6px;
            font-weight: 600;
            background: #252b35;
        }
        QToolButton#SectionHeader:hover { background: #303746; }
        QToolButton#SectionHeader:checked {
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;
        }
        QFrame#SectionContent {
            border: 1px solid #3a4250;
            border-top: 0;
            border-bottom-left-radius: 6px;
            border-bottom-right-radius: 6px;
            background: #20242b;
        }
        QPushButton { padding: 7px 10px; border-radius: 4px; }
        QPushButton:hover { background: #3a4250; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            padding: 4px;
            border: 1px solid #465161;
            border-radius: 4px;
            background: #15181d;
        }
        QTableWidget {
            gridline-color: #3a4250;
            selection-background-color: #4c8bf5;
        }
        QLabel#PanelTitle { font-size: 16pt; font-weight: 700; }
        """
    )


def _double_box(
    value: float,
    minimum: float = -1.0e6,
    maximum: float = 1.0e6,
    step: float = 1.0,
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(6)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def _angle_box(value: float) -> QDoubleSpinBox:
    box = _double_box(value, minimum=-360.0, maximum=360.0, step=1.0)
    box.setSuffix(" deg")
    return box


def _view_model(model: ModalModel, labels: list[int]) -> ModalModel:
    return ModalModel(
        path=model.path,
        blocks=[],
        nodes={label: model.nodes[label] for label in labels if label in model.nodes},
        elements=model.elements,
        trace_lines=model.trace_lines,
        coordinate_systems=model.coordinate_systems,
        units=model.units,
        modes=model.modes,
        functions=model.functions,
        diagnostics=model.diagnostics,
        metadata=model.metadata,
    )


def _scalar_bar_args(title: str, position: str = "Left") -> dict[str, object]:
    left = position != "Right"
    return {
        "title": title,
        "vertical": True,
        "position_x": 0.02 if left else 0.90,
        "position_y": 0.14,
        "width": 0.08,
        "height": 0.72,
        "title_font_size": 12,
        "label_font_size": 10,
        "color": "white",
    }


def _make_point_picker() -> object | None:
    try:
        from vtkmodules.vtkRenderingCore import vtkPointPicker

        picker = vtkPointPicker()
        picker.SetTolerance(0.025)
        return picker
    except Exception:
        return None


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


class _NullPlotter(QWidget):
    """Small test double for Windows offscreen tests where VTK can crash natively."""

    renderer = None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mesh_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.clear_count = 0
        self.render_count = 0

    def clear(self) -> None:
        self.clear_count += 1
        self.mesh_calls.clear()
        return None

    def set_background(self, color: str) -> None:
        return None

    def add_axes(self) -> None:
        return None

    def add_text(self, *args: object, **kwargs: object) -> None:
        return None

    def render(self) -> None:
        self.render_count += 1
        return None

    def add_mesh(self, *args: object, **kwargs: object) -> None:
        self.mesh_calls.append((args, kwargs))
        return None

    def reset_camera(self) -> None:
        return None

    def screenshot(self, path: str | None = None, *args: object, **kwargs: object) -> np.ndarray | None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        if path:
            Path(path).write_bytes(b"")
        if kwargs.get("return_img", True):
            return image
        return None
