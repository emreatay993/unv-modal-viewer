from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import ModalModel, TransformSpec


class ModeNormalization:
    RAW = "raw"
    MAX_DISPLACEMENT = "max_displacement"
    UNIT_MODAL_MASS = "unit_modal_mass"

    LABELS = {
        RAW: "Raw",
        MAX_DISPLACEMENT: "Max displacement",
        UNIT_MODAL_MASS: "Unit modal mass",
    }

    BY_LABEL = {label: key for key, label in LABELS.items()}


@dataclass(slots=True)
class RenderOptions:
    colormap: str = "viridis"
    reverse_colormap: bool = False
    scalar_auto: bool = True
    scalar_min: float = 0.0
    scalar_max: float = 1.0
    legend_visible: bool = True
    legend_position: str = "Left"
    surface_opacity: float = 0.58
    point_size: int = 10
    selected_color: str = "#ffd166"

    @property
    def pyvista_colormap(self) -> str:
        return f"{self.colormap}_r" if self.reverse_colormap else self.colormap

    @property
    def clim(self) -> tuple[float, float] | None:
        if self.scalar_auto:
            return None
        low, high = float(self.scalar_min), float(self.scalar_max)
        if low == high:
            high = low + 1.0
        return (min(low, high), max(low, high))


@dataclass(slots=True)
class SelectionState:
    selected_node_ids: set[int] = field(default_factory=set)
    hidden_node_ids: set[int] = field(default_factory=set)
    isolate_selected: bool = False

    def clear_selection(self) -> None:
        self.selected_node_ids.clear()

    def select_only(self, node_id: int) -> None:
        self.selected_node_ids = {int(node_id)}

    def toggle(self, node_id: int) -> None:
        node = int(node_id)
        if node in self.selected_node_ids:
            self.selected_node_ids.remove(node)
        else:
            self.selected_node_ids.add(node)

    def invert(self, labels: list[int]) -> None:
        all_labels = set(labels)
        self.selected_node_ids = all_labels - self.selected_node_ids

    def hide_selected(self) -> None:
        self.hidden_node_ids.update(self.selected_node_ids)

    def show_all(self) -> None:
        self.hidden_node_ids.clear()
        self.isolate_selected = False

    def visible_labels(self, labels: list[int]) -> list[int]:
        if self.isolate_selected and self.selected_node_ids:
            return [label for label in labels if label in self.selected_node_ids and label not in self.hidden_node_ids]
        return [label for label in labels if label not in self.hidden_node_ids]


@dataclass(slots=True)
class OverlayState:
    model: ModalModel | None = None
    path: Path | None = None
    visible: bool = True
    transform: TransformSpec = field(default_factory=TransformSpec.identity)
    opacity: float = 0.32
    color: str = "#f59e0b"
    show_points: bool = True
    show_surface: bool = True
    mode_index: int = -1
    deformation_scale: float = 1.0

    def selected_mode(self):
        if self.model is None or self.mode_index < 0 or self.mode_index >= len(self.model.modes):
            return None
        return self.model.modes[self.mode_index]


@dataclass(slots=True)
class MacOptions:
    components: str = "XYZ"
    use_nearest_fallback: bool = False
    nearest_tolerance: float = 1.0e-6


def color_choices() -> dict[str, str]:
    return {
        "Gold": "#ffd166",
        "Orange": "#f59e0b",
        "Red": "#ef4444",
        "Green": "#22c55e",
        "Cyan": "#06b6d4",
        "Blue": "#3b82f6",
        "White": "#ffffff",
    }


def transform_from_values(
    scale: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    origin: np.ndarray | None = None,
) -> TransformSpec:
    return TransformSpec(
        scale=np.asarray(scale, dtype=float),
        translation=np.asarray(translation, dtype=float),
        cs_rotation=np.asarray(rotation, dtype=float),
        cs_origin=np.zeros(3, dtype=float) if origin is None else np.asarray(origin, dtype=float),
    )
