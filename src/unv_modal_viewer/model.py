from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class RawDatasetBlock:
    index: int
    dataset_type: int
    start_marker: str
    type_line: str
    content_lines: list[str]
    end_marker: str

    def original_lines(self) -> list[str]:
        return [self.start_marker, self.type_line, *self.content_lines, self.end_marker]


@dataclass(slots=True)
class Node:
    label: int
    coordinates: np.ndarray
    definition_cs: int = 0
    displacement_cs: int = 0
    color: int = 0
    source_dataset: int = 0
    block_index: int = -1


@dataclass(slots=True)
class Element:
    label: int
    descriptor: int
    node_labels: list[int]
    color: int = 0
    block_index: int = -1


@dataclass(slots=True)
class TraceLine:
    label: int
    node_labels: list[int]
    color: int = 0
    name: str = ""
    block_index: int = -1


@dataclass(slots=True)
class CoordinateSystem:
    label: int
    name: str
    kind: int
    rotation: np.ndarray
    origin: np.ndarray
    color: int = 0
    block_index: int = -1


@dataclass(slots=True)
class Units:
    code: int
    description: str
    temperature_mode: int
    factors_to_si_divisor: tuple[float, float, float, float]
    block_index: int = -1


@dataclass(slots=True)
class Header:
    model_name: str
    description: str
    db_app: str
    date_db_created: str
    time_db_created: str
    version_db1: int | None
    version_db2: int | None
    file_type: int | None
    date_db_saved: str
    time_db_saved: str
    program: str
    date_file_written: str
    time_file_written: str
    block_index: int = -1


@dataclass(slots=True)
class ModeShape:
    name: str
    source_dataset: int
    block_index: int
    mode_number: int | None
    frequency_hz: float | None
    modal_mass: float | None
    viscous_damping: float | None
    hysteretic_damping: float | None
    data_characteristic: int
    result_type: int
    data_type: int
    ndv: int
    node_values: dict[int, np.ndarray]
    id_lines: list[str] = field(default_factory=list)

    @property
    def component_names(self) -> list[str]:
        if self.ndv >= 6:
            return ["X", "Y", "Z", "Rx", "Ry", "Rz"][: self.ndv]
        if self.ndv == 3:
            return ["X", "Y", "Z"]
        return [f"C{i + 1}" for i in range(self.ndv)]


@dataclass(slots=True)
class FunctionSummary:
    block_index: int
    function_type: int | None
    function_id: int | None
    response_node: int | None
    response_direction: int | None
    reference_node: int | None
    reference_direction: int | None
    n_values: int | None
    x_min: float | None
    x_increment: float | None
    description: str = ""


@dataclass(slots=True)
class TransformSpec:
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=float))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    cs_rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    cs_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    @classmethod
    def identity(cls) -> "TransformSpec":
        return cls()


@dataclass(slots=True)
class ModalModel:
    path: Path | None
    blocks: list[RawDatasetBlock]
    nodes: dict[int, Node] = field(default_factory=dict)
    elements: list[Element] = field(default_factory=list)
    trace_lines: list[TraceLine] = field(default_factory=list)
    coordinate_systems: dict[int, CoordinateSystem] = field(default_factory=dict)
    header: Header | None = None
    units: Units | None = None
    modes: list[ModeShape] = field(default_factory=list)
    functions: list[FunctionSummary] = field(default_factory=list)
    raw_prefix: list[str] = field(default_factory=list)
    raw_suffix: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_labels(self) -> list[int]:
        return sorted(self.nodes)

    @property
    def points(self) -> np.ndarray:
        labels = self.node_labels
        if not labels:
            return np.empty((0, 3), dtype=float)
        return np.vstack([self.nodes[label].coordinates for label in labels]).astype(float)

    def node_index(self) -> dict[int, int]:
        return {label: i for i, label in enumerate(self.node_labels)}
