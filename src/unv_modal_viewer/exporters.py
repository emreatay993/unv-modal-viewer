from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .modal_analysis import mode_label
from .model import ModalModel, ModeShape, TransformSpec
from .transforms import transformed_node_coordinates
from .visualization import element_surface, point_cloud

SUPPORTED_ANIMATION_EXTENSIONS = {".mp4", ".avi", ".gif"}


def export_nodes_csv(
    model: ModalModel,
    path: str | Path,
    transform: TransformSpec | None = None,
    node_labels: Iterable[int] | None = None,
    scalars: dict[int, float] | None = None,
) -> None:
    labels = list(model.node_labels if node_labels is None else node_labels)
    transformed = transformed_node_coordinates(model, transform or TransformSpec.identity())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "x", "y", "z", "transformed_x", "transformed_y", "transformed_z", "scalar"])
        for label in labels:
            if label not in model.nodes:
                continue
            original = model.nodes[label].coordinates
            current = transformed.get(label, original)
            writer.writerow(
                [
                    label,
                    *[f"{value:.12g}" for value in original],
                    *[f"{value:.12g}" for value in current],
                    "" if scalars is None or label not in scalars else f"{scalars[label]:.12g}",
                ]
            )


def export_modes_csv(
    model: ModalModel,
    path: str | Path,
    modes: Iterable[ModeShape] | None = None,
    node_labels: Iterable[int] | None = None,
) -> None:
    labels = list(model.node_labels if node_labels is None else node_labels)
    selected_modes = list(model.modes if modes is None else modes)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode_index",
                "mode",
                "frequency_hz",
                "modal_mass",
                "viscous_damping",
                "hysteretic_damping",
                "node_id",
                "component",
                "real",
                "imaginary",
            ]
        )
        for mode_index, mode in enumerate(selected_modes):
            for label in labels:
                values = mode.node_values.get(label)
                if values is None:
                    continue
                for component, real, imaginary in _component_rows(mode, values):
                    writer.writerow(
                        [
                            mode_index,
                            mode.mode_number if mode.mode_number is not None else "",
                            "" if mode.frequency_hz is None else f"{mode.frequency_hz:.12g}",
                            "" if mode.modal_mass is None else f"{mode.modal_mass:.12g}",
                            "" if mode.viscous_damping is None else f"{mode.viscous_damping:.12g}",
                            "" if mode.hysteretic_damping is None else f"{mode.hysteretic_damping:.12g}",
                            label,
                            component,
                            f"{real:.12g}",
                            f"{imaginary:.12g}",
                        ]
                    )


def export_mac_csv(
    path: str | Path,
    matrix: np.ndarray,
    row_modes: list[ModeShape],
    column_modes: list[ModeShape],
) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", *[mode_label(mode, index) for index, mode in enumerate(column_modes)]])
        for row, mode in enumerate(row_modes):
            writer.writerow([mode_label(mode, row), *[f"{float(value):.12g}" for value in matrix[row]]])


def export_scene_vtk(
    model: ModalModel,
    path: str | Path,
    points: np.ndarray,
    scalars: np.ndarray,
    selected_labels: set[int] | None = None,
    hidden_labels: set[int] | None = None,
) -> None:
    selected = selected_labels or set()
    hidden = hidden_labels or set()
    mesh = element_surface(model, points)
    if mesh is None:
        mesh = point_cloud(model, points)
    labels = np.array(model.node_labels, dtype=int)
    mesh.point_data["node_id"] = labels
    mesh.point_data["value"] = np.asarray(scalars, dtype=float)
    mesh.point_data["selected"] = np.array([1 if int(label) in selected else 0 for label in labels], dtype=int)
    mesh.point_data["hidden"] = np.array([1 if int(label) in hidden else 0 for label in labels], dtype=int)
    mesh.save(str(path))


def export_screenshot(plotter: object, path: str | Path) -> None:
    plotter.screenshot(str(path))


def export_animation_media(
    path: str | Path,
    frame_source: Callable[[float], np.ndarray],
    duration_seconds: float,
    fps: int,
    writer_factory: Callable[[Path, int], object] | None = None,
) -> int:
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in SUPPORTED_ANIMATION_EXTENSIONS:
        raise ValueError("Animation export supports .mp4, .avi, and .gif files.")

    safe_fps = max(1, int(fps))
    frame_count = max(1, int(round(max(0.1, float(duration_seconds)) * safe_fps)))
    writer_factory = writer_factory or _imageio_writer

    with writer_factory(destination, safe_fps) as writer:
        for frame_index in range(frame_count):
            phase = float(np.sin(2.0 * np.pi * frame_index / frame_count))
            writer.append_data(_rgb_uint8(frame_source(phase)))
    return frame_count


def _imageio_writer(path: Path, fps: int) -> object:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Install imageio and imageio-ffmpeg to export animation media.") from exc
    return imageio.get_writer(str(path), fps=fps)


def _rgb_uint8(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Animation frames must be grayscale, RGB, or RGBA images.")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _component_rows(mode: ModeShape, values: np.ndarray) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    names = mode.component_names
    if mode.data_type in (5, 6):
        real = values[0::2]
        imag = values[1::2]
        for index, value in enumerate(real):
            rows.append((_component_name(names, index), float(value), float(imag[index] if index < len(imag) else 0.0)))
        return rows

    for index, value in enumerate(values):
        rows.append((_component_name(names, index), float(value), 0.0))
    return rows


def _component_name(names: list[str], index: int) -> str:
    return names[index] if index < len(names) else f"C{index + 1}"
