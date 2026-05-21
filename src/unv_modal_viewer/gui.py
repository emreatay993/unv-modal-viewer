from __future__ import annotations

import os
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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyleFactory,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from .io import export_unv, load_unv
from .model import CoordinateSystem, ModalModel, ModeShape, TransformSpec
from .transforms import transformed_node_coordinates
from .visualization import (
    deformed_points,
    element_surface,
    generated_surface,
    point_cloud,
    supported_surface_elements,
    trace_line_mesh,
)


class MainWindow(QMainWindow):
    def __init__(self, initial_path: str | Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("UNV Modal Test Viewer")
        self.resize(1440, 900)

        self.model: ModalModel | None = None
        self.current_path: Path | None = None
        self._current_points = np.empty((0, 3), dtype=float)
        self._current_scalars = np.empty(0, dtype=float)
        self._current_labels: list[int] = []
        self._phase = 1.0
        self._hover_observer_installed = False
        self._point_picker = _make_point_picker()

        self._build_ui()
        self._connect_signals()
        self._install_hover_observer()
        self.refresh_scene(reset_camera=True)

        if initial_path:
            self.load_file(Path(initial_path))

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())

        self.plotter = _NullPlotter(self) if os.getenv("UNV_MODAL_VIEWER_TEST_NO_VTK") else QtInteractor(self)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.plotter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1020])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

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

        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
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
        self.summary_label = QLabel("Datasets: none")
        self.summary_label.setWordWrap(True)
        file_layout.addWidget(self.summary_label)
        layout.addWidget(file_group)

        transform_group = QGroupBox("Coordinate Transform")
        transform_layout = QFormLayout(transform_group)
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
        layout.addWidget(transform_group)

        axes_group = QGroupBox("CS Rotation Rows")
        axes_layout = QGridLayout(axes_group)
        self.axis_boxes: list[QDoubleSpinBox] = []
        defaults = np.eye(3)
        for row in range(3):
            axes_layout.addWidget(QLabel(f"R{row + 1}"), row, 0)
            for col in range(3):
                box = _double_box(float(defaults[row, col]), minimum=-1.0, maximum=1.0, step=0.05)
                self.axis_boxes.append(box)
                axes_layout.addWidget(box, row, col + 1)
        layout.addWidget(axes_group)

        view_group = QGroupBox("View")
        view_layout = QFormLayout(view_group)
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
        self.point_size = QSpinBox()
        self.point_size.setRange(2, 30)
        self.point_size.setValue(10)
        view_layout.addRow("Point size", self.point_size)
        layout.addWidget(view_group)

        mode_group = QGroupBox("Modes")
        mode_layout = QVBoxLayout(mode_group)
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
        self.deformation_scale = _double_box(1.0, minimum=-1.0e9, maximum=1.0e9, step=0.1)
        self.animate_mode = QCheckBox()
        mode_controls.addRow("Color by", self.component_combo)
        mode_controls.addRow("Deformation scale", self.deformation_scale)
        mode_controls.addRow("Animate", self.animate_mode)
        mode_layout.addLayout(mode_controls)
        layout.addWidget(mode_group)

        layout.addStretch(1)
        return container

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(self._choose_file)
        self.export_button.clicked.connect(self._export_file)
        self.cs_combo.currentIndexChanged.connect(self._coordinate_system_selected)
        self.mode_table.itemSelectionChanged.connect(lambda: self.refresh_scene(reset_camera=False))
        self.component_combo.currentIndexChanged.connect(lambda: self.refresh_scene(reset_camera=False))
        self.animate_mode.toggled.connect(self._animation_toggled)

        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(60)
        self.animation_timer.timeout.connect(self._animation_tick)

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
        for box in self.axis_boxes:
            box.valueChanged.connect(lambda *_: self.refresh_scene(reset_camera=False))
        for box in [self.show_points, self.show_surface, self.generate_surface, self.show_traces]:
            box.toggled.connect(lambda *_: self.refresh_scene(reset_camera=False))

    def load_file(self, path: Path) -> None:
        try:
            self.model = load_unv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.current_path = path
        self.path_label.setText(str(path))
        self.export_button.setEnabled(True)
        self._populate_coordinate_systems()
        self._populate_modes()
        self._update_summary()
        self.statusBar().showMessage(f"Loaded {path.name}")
        self.refresh_scene(reset_camera=True)

    def refresh_scene(self, reset_camera: bool = False) -> None:
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

        model = self.model
        spec = self.current_transform()
        transformed = transformed_node_coordinates(model, spec)
        base_points = np.vstack([transformed[label] for label in model.node_labels])
        mode = self.selected_mode()
        phase = self._phase if self.animate_mode.isChecked() else 1.0
        points, scalars = deformed_points(
            model,
            mode,
            self.deformation_scale.value() * phase,
            self.component_combo.currentText(),
            base_points=base_points,
        )
        if mode is None:
            scalars = np.array(model.node_labels, dtype=float)

        self._current_points = points
        self._current_scalars = scalars
        self._current_labels = model.node_labels

        scalar_title = self.component_combo.currentText() if mode is not None else "Node"
        surface = None
        if self.show_surface.isChecked():
            surface = element_surface(model, points)
        if surface is None and self.generate_surface.isChecked():
            surface = generated_surface(model, points)
        if surface is not None:
            surface.point_data["value"] = scalars
            self.plotter.add_mesh(
                surface,
                scalars="value",
                cmap="viridis",
                opacity=0.58,
                smooth_shading=True,
                scalar_bar_args=_left_scalar_bar_args(scalar_title),
            )

        if self.show_traces.isChecked():
            traces = trace_line_mesh(model, points)
            if traces is not None:
                self.plotter.add_mesh(traces, color="#d8dee9", line_width=3, render_lines_as_tubes=True)

        if self.show_points.isChecked():
            cloud = point_cloud(model, points)
            cloud.point_data["value"] = scalars
            self.plotter.add_mesh(
                cloud,
                scalars="value",
                cmap="viridis",
                point_size=self.point_size.value(),
                render_points_as_spheres=True,
                scalar_bar_args=_left_scalar_bar_args(scalar_title),
                show_scalar_bar=surface is None,
            )

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

    def current_transform(self) -> TransformSpec:
        rotation = np.array([box.value() for box in self.axis_boxes], dtype=float).reshape(3, 3)
        return TransformSpec(
            scale=np.array([self.scale_x.value(), self.scale_y.value(), self.scale_z.value()]),
            translation=np.array([self.trans_x.value(), self.trans_y.value(), self.trans_z.value()]),
            cs_rotation=rotation,
            cs_origin=np.array([self.origin_x.value(), self.origin_y.value(), self.origin_z.value()]),
        )

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
            str(self.current_path.parent if self.current_path else Path.home()),
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

    def _coordinate_system_selected(self) -> None:
        label = self.cs_combo.currentData()
        self._set_coordinate_system(None if label is None or self.model is None else self.model.coordinate_systems[label])
        self.refresh_scene(reset_camera=False)

    def _set_coordinate_system(self, cs: CoordinateSystem | None) -> None:
        rotation = np.eye(3) if cs is None else cs.rotation
        origin = np.zeros(3) if cs is None else cs.origin
        for box, value in zip(self.axis_boxes, rotation.reshape(-1), strict=False):
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)
        for box, value in zip([self.origin_x, self.origin_y, self.origin_z], origin, strict=False):
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)

    def _animation_toggled(self, enabled: bool) -> None:
        self._phase = 0.0 if enabled else 1.0
        if enabled:
            self.animation_timer.start()
        else:
            self.animation_timer.stop()
            self.refresh_scene(reset_camera=False)

    def _animation_tick(self) -> None:
        self._phase = float(np.sin(np.arcsin(np.clip(self._phase, -1.0, 1.0)) + 0.16))
        if abs(self._phase) > 0.99:
            self._phase *= -0.95
        self.refresh_scene(reset_camera=False)

    def _install_hover_observer(self) -> None:
        if self._hover_observer_installed or self._point_picker is None:
            return
        try:
            self.plotter.iren.add_observer("MouseMoveEvent", self._on_mouse_move)
            self._hover_observer_installed = True
        except Exception:
            try:
                self.plotter.iren.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
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
        QGroupBox {
            border: 1px solid #3a4250;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 12px;
            font-weight: 600;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
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


def _left_scalar_bar_args(title: str) -> dict[str, object]:
    return {
        "title": title,
        "vertical": True,
        "position_x": 0.02,
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

    def clear(self) -> None:
        return None

    def set_background(self, color: str) -> None:
        return None

    def add_axes(self) -> None:
        return None

    def add_text(self, *args: object, **kwargs: object) -> None:
        return None

    def render(self) -> None:
        return None

    def add_mesh(self, *args: object, **kwargs: object) -> None:
        return None

    def reset_camera(self) -> None:
        return None
