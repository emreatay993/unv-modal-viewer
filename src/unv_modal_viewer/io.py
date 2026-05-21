from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .model import (
    CoordinateSystem,
    Element,
    FunctionSummary,
    ModalModel,
    ModeShape,
    Node,
    RawDatasetBlock,
    TraceLine,
    TransformSpec,
    Units,
)
from .transforms import transformed_mode_shape, transformed_node_coordinates


SUPPORTED_DATASETS = {15, 55, 58, 82, 164, 2411, 2412, 2414, 2420}
BEAM_DESCRIPTOR_IDS = {11, 21, 22, 23, 24}


def load_unv(path: str | Path) -> ModalModel:
    file_path = Path(path)
    lines = file_path.read_text(encoding="latin-1").splitlines(keepends=True)
    blocks, prefix, suffix = _split_blocks(lines)
    model = ModalModel(path=file_path, blocks=blocks, raw_prefix=prefix, raw_suffix=suffix)

    for block in blocks:
        try:
            _parse_supported_block(model, block)
        except Exception as exc:  # pragma: no cover - defensive diagnostic path
            model.diagnostics.append(
                f"Dataset {block.dataset_type} at block {block.index} was preserved but not fully parsed: {exc}"
            )

    _augment_with_pyuff_metadata(model, file_path)
    model.metadata["dataset_counts"] = dict(Counter(block.dataset_type for block in blocks))
    return model


def export_unv(
    model: ModalModel,
    path: str | Path,
    transform: TransformSpec | None = None,
    transform_vectors: bool = False,
) -> None:
    spec = transform or TransformSpec.identity()
    transformed_nodes = transformed_node_coordinates(model, spec)
    modes_by_block = {mode.block_index: mode for mode in model.modes}
    output: list[str] = [*model.raw_prefix]

    for block in model.blocks:
        if block.dataset_type == 2411:
            output.extend(_rewrite_2411(block, transformed_nodes))
        elif block.dataset_type == 15:
            output.extend(_rewrite_15(block, transformed_nodes))
        elif transform_vectors and block.dataset_type == 55 and block.index in modes_by_block:
            output.extend(_rewrite_55(block, modes_by_block[block.index], spec))
        elif transform_vectors and block.dataset_type == 2414 and block.index in modes_by_block:
            output.extend(_rewrite_2414(block, modes_by_block[block.index], spec))
        else:
            output.extend(block.original_lines())

    output.extend(model.raw_suffix)
    Path(path).write_text("".join(output), encoding="latin-1")


def _split_blocks(lines: list[str]) -> tuple[list[RawDatasetBlock], list[str], list[str]]:
    blocks: list[RawDatasetBlock] = []
    prefix: list[str] = []
    suffix: list[str] = []
    i = 0

    while i < len(lines):
        if lines[i].strip() != "-1":
            (suffix if blocks else prefix).append(lines[i])
            i += 1
            continue

        if i + 1 >= len(lines):
            (suffix if blocks else prefix).append(lines[i])
            i += 1
            continue

        try:
            dataset_type = int(lines[i + 1].strip())
        except ValueError:
            (suffix if blocks else prefix).append(lines[i])
            i += 1
            continue

        j = i + 2
        while j < len(lines) and lines[j].strip() != "-1":
            j += 1

        end_marker = lines[j] if j < len(lines) else _line("    -1")
        blocks.append(
            RawDatasetBlock(
                index=len(blocks),
                dataset_type=dataset_type,
                start_marker=lines[i],
                type_line=lines[i + 1],
                content_lines=lines[i + 2 : j],
                end_marker=end_marker,
            )
        )
        i = j + 1

    return blocks, prefix, suffix


def _parse_supported_block(model: ModalModel, block: RawDatasetBlock) -> None:
    if block.dataset_type in (15, 2411):
        _parse_nodes(model, block)
    elif block.dataset_type == 82:
        model.trace_lines.extend(_parse_trace_lines(block))
    elif block.dataset_type == 164:
        model.units = _parse_units(block)
    elif block.dataset_type == 2412:
        model.elements.extend(_parse_elements(block))
    elif block.dataset_type == 2420:
        model.coordinate_systems.update(_parse_coordinate_systems(block))
    elif block.dataset_type == 55:
        mode = _parse_55(block)
        if mode is not None:
            model.modes.append(mode)
    elif block.dataset_type == 2414:
        mode = _parse_2414(block)
        if mode is not None:
            model.modes.append(mode)
    elif block.dataset_type == 58:
        model.functions.append(_parse_58_summary(block))


def _parse_nodes(model: ModalModel, block: RawDatasetBlock) -> None:
    lines = block.content_lines
    if block.dataset_type == 2411:
        for i in range(0, len(lines) - 1, 2):
            ints = _ints(lines[i])
            xyz = _floats(lines[i + 1])
            if len(ints) < 4 or len(xyz) < 3:
                continue
            node = Node(
                label=ints[0],
                definition_cs=ints[1],
                displacement_cs=ints[2],
                color=ints[3],
                coordinates=np.array(xyz[:3], dtype=float),
                source_dataset=2411,
                block_index=block.index,
            )
            _store_node(model, node)
        return

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            ints = [int(value) for value in parts[:4]]
            xyz = [_float(value) for value in parts[4:7]]
        except ValueError:
            continue
        node = Node(
            label=ints[0],
            definition_cs=ints[1],
            displacement_cs=ints[2],
            color=ints[3],
            coordinates=np.array(xyz, dtype=float),
            source_dataset=15,
            block_index=block.index,
        )
        _store_node(model, node)


def _store_node(model: ModalModel, node: Node) -> None:
    existing = model.nodes.get(node.label)
    if existing is None or node.source_dataset == 2411 or existing.source_dataset != 2411:
        model.nodes[node.label] = node


def _parse_trace_lines(block: RawDatasetBlock) -> list[TraceLine]:
    traces: list[TraceLine] = []
    i = 0
    lines = block.content_lines
    while i < len(lines):
        header = _ints(lines[i])
        if len(header) < 3:
            i += 1
            continue
        label, count, color = header[:3]
        name = lines[i + 1].strip() if i + 1 < len(lines) else ""
        i += 2
        raw_nodes: list[int] = []
        while i < len(lines) and len(raw_nodes) < count:
            raw_nodes.extend(_ints(lines[i]))
            i += 1
        nodes = [abs(node) for node in raw_nodes if node != 0][:count]
        traces.append(TraceLine(label, nodes, color=color, name=name, block_index=block.index))
    return traces


def _parse_units(block: RawDatasetBlock) -> Units | None:
    if not block.content_lines:
        return None
    first = block.content_lines[0].strip()
    parts = first.split()
    try:
        code = int(parts[0])
        temperature_mode = int(parts[-1])
        description = " ".join(parts[1:-1]).strip()
    except (ValueError, IndexError):
        code = 0
        temperature_mode = 0
        description = first

    factors: list[float] = []
    for line in block.content_lines[1:]:
        factors.extend(_floats(line))
        if len(factors) >= 4:
            break
    while len(factors) < 4:
        factors.append(1.0 if len(factors) < 3 else 0.0)
    return Units(code, description, temperature_mode, tuple(factors[:4]), block_index=block.index)


def _parse_elements(block: RawDatasetBlock) -> list[Element]:
    elements: list[Element] = []
    i = 0
    lines = block.content_lines
    while i < len(lines):
        header = _ints(lines[i])
        if len(header) < 6:
            i += 1
            continue
        label, descriptor, _, _, color, node_count = header[:6]
        i += 1
        if descriptor in BEAM_DESCRIPTOR_IDS and i < len(lines):
            i += 1
        nodes: list[int] = []
        while i < len(lines) and len(nodes) < node_count:
            nodes.extend(_ints(lines[i]))
            i += 1
        elements.append(
            Element(label, descriptor, nodes[:node_count], color=color, block_index=block.index)
        )
    return elements


def _parse_coordinate_systems(block: RawDatasetBlock) -> dict[int, CoordinateSystem]:
    systems: dict[int, CoordinateSystem] = {}
    lines = block.content_lines
    i = 2 if len(lines) >= 2 else 0
    while i + 5 < len(lines):
        header = _ints(lines[i])
        if len(header) < 3:
            i += 1
            continue
        label, kind, color = header[:3]
        name = lines[i + 1].strip()
        rows = [_floats(lines[i + offset]) for offset in range(2, 6)]
        if any(len(row) < 3 for row in rows):
            i += 1
            continue
        rotation = np.array([row[:3] for row in rows[:3]], dtype=float)
        origin = np.array(rows[3][:3], dtype=float)
        systems[label] = CoordinateSystem(
            label=label,
            name=name,
            kind=kind,
            color=color,
            rotation=rotation,
            origin=origin,
            block_index=block.index,
        )
        i += 6
    return systems


def _parse_55(block: RawDatasetBlock) -> ModeShape | None:
    lines = block.content_lines
    if len(lines) < 8:
        return None

    id_lines = [line.rstrip("\r\n") for line in lines[:5]]
    rec6 = _ints(lines[5])
    rec7 = _ints(lines[6])
    rec8 = _floats(lines[7])
    if len(rec6) < 6:
        return None

    analysis_type = rec6[1]
    data_type = rec6[4]
    ndv = rec6[5]
    needed = ndv * (2 if data_type in (5, 6) else 1)
    node_values = _parse_node_value_records(lines[8:], needed)
    if not node_values:
        return None

    mode_number = rec7[3] if len(rec7) > 3 else None
    frequency = rec8[0] if len(rec8) > 0 else None
    modal_mass = rec8[1] if len(rec8) > 1 else None
    viscous = rec8[2] if len(rec8) > 2 else None
    hysteretic = rec8[3] if len(rec8) > 3 else None
    name = id_lines[0].strip() or f"Dataset 55 block {block.index}"
    if analysis_type == 2 and mode_number is not None:
        name = f"Mode {mode_number}"

    return ModeShape(
        name=name,
        source_dataset=55,
        block_index=block.index,
        mode_number=mode_number,
        frequency_hz=frequency,
        modal_mass=modal_mass,
        viscous_damping=viscous,
        hysteretic_damping=hysteretic,
        data_characteristic=rec6[2],
        result_type=rec6[3],
        data_type=data_type,
        ndv=ndv,
        node_values=node_values,
        id_lines=id_lines,
    )


def _parse_2414(block: RawDatasetBlock) -> ModeShape | None:
    lines = block.content_lines
    if len(lines) < 14:
        return None
    label = _first_int(lines[0])
    location = _first_int(lines[2])
    if location != 1:
        return None

    id_lines = [line.rstrip("\r\n") for line in lines[3:8]]
    rec9 = _ints(lines[8])
    if len(rec9) < 6:
        return None
    int_params = _ints(lines[9]) + _ints(lines[10])
    real_params = _floats(lines[11]) + _floats(lines[12])
    data_type = rec9[4]
    ndv = rec9[5]
    needed = ndv * (2 if data_type in (5, 6) else 1)
    node_values = _parse_node_value_records(lines[13:], needed)
    if not node_values:
        return None

    mode_number = _mode_number_from_2414(label, int_params)
    frequency, modal_mass, viscous, hysteretic = _normal_mode_reals_from_2414(real_params)
    name = id_lines[0].strip() or f"Dataset 2414 block {block.index}"
    if rec9[1] == 2 and mode_number is not None:
        name = f"Mode {mode_number}"

    return ModeShape(
        name=name,
        source_dataset=2414,
        block_index=block.index,
        mode_number=mode_number,
        frequency_hz=frequency,
        modal_mass=modal_mass,
        viscous_damping=viscous,
        hysteretic_damping=hysteretic,
        data_characteristic=rec9[2],
        result_type=rec9[3],
        data_type=data_type,
        ndv=ndv,
        node_values=node_values,
        id_lines=id_lines,
    )


def _parse_node_value_records(lines: list[str], needed: int) -> dict[int, np.ndarray]:
    values_by_node: dict[int, np.ndarray] = {}
    i = 0
    while i < len(lines):
        node = _first_int(lines[i])
        if node is None:
            i += 1
            continue
        i += 1
        values: list[float] = []
        while i < len(lines) and len(values) < needed:
            if _looks_like_node_label_line(lines[i]):
                break
            floats = _floats(lines[i])
            if not floats and _first_int(lines[i]) is not None:
                break
            values.extend(floats)
            i += 1
        if len(values) >= needed:
            values_by_node[node] = np.array(values[:needed], dtype=float)
    return values_by_node


def _parse_58_summary(block: RawDatasetBlock) -> FunctionSummary:
    lines = block.content_lines
    description = lines[0].strip() if lines else ""
    rec6 = lines[5].split() if len(lines) > 5 else []
    rec7 = lines[6].split() if len(lines) > 6 else []
    return FunctionSummary(
        block_index=block.index,
        function_type=_safe_int(rec6, 0),
        function_id=_safe_int(rec6, 1),
        response_node=_safe_int(rec6, 5),
        response_direction=_safe_int(rec6, 6),
        reference_node=_safe_int(rec6, 8),
        reference_direction=_safe_int(rec6, 9),
        n_values=_safe_int(rec7, 1),
        x_min=_safe_float(rec7, 3),
        x_increment=_safe_float(rec7, 4),
        description=description,
    )


def _augment_with_pyuff_metadata(model: ModalModel, path: Path) -> None:
    try:
        import pyuff  # type: ignore

        uff = pyuff.UFF(str(path))
        model.metadata["pyuff_set_types"] = [int(value) for value in uff.get_set_types()]
        model.metadata["pyuff_set_formats"] = [int(value) for value in uff.get_set_formats()]
    except Exception as exc:
        model.diagnostics.append(f"pyuff metadata read skipped: {exc}")


def _rewrite_2411(block: RawDatasetBlock, transformed_nodes: dict[int, np.ndarray]) -> list[str]:
    lines = [block.start_marker, block.type_line]
    content = block.content_lines
    i = 0
    while i < len(content) - 1:
        ints = _ints(content[i])
        xyz = _floats(content[i + 1])
        if len(ints) < 4 or len(xyz) < 3:
            lines.extend([content[i], content[i + 1]])
            i += 2
            continue
        coords = transformed_nodes.get(ints[0], np.array(xyz[:3], dtype=float))
        lines.append(f"{ints[0]:10d}{ints[1]:10d}{ints[2]:10d}{ints[3]:10d}\n")
        lines.append(_format_floats(coords, width=25, precision=16, per_line=3))
        i += 2
    if i < len(content):
        lines.append(content[i])
    lines.append(block.end_marker)
    return lines


def _rewrite_15(block: RawDatasetBlock, transformed_nodes: dict[int, np.ndarray]) -> list[str]:
    lines = [block.start_marker, block.type_line]
    for content_line in block.content_lines:
        parts = content_line.split()
        if len(parts) < 7:
            lines.append(content_line)
            continue
        try:
            ints = [int(value) for value in parts[:4]]
            original = np.array([_float(value) for value in parts[4:7]], dtype=float)
        except ValueError:
            lines.append(content_line)
            continue
        coords = transformed_nodes.get(ints[0], original)
        lines.append(
            f"{ints[0]:10d}{ints[1]:10d}{ints[2]:10d}{ints[3]:10d}"
            f"{coords[0]:13.5E}{coords[1]:13.5E}{coords[2]:13.5E}\n"
        )
    lines.append(block.end_marker)
    return lines


def _rewrite_55(block: RawDatasetBlock, mode: ModeShape, spec: TransformSpec) -> list[str]:
    transformed = transformed_mode_shape(mode, spec)
    header = block.content_lines[:8]
    return [
        block.start_marker,
        block.type_line,
        *header,
        *_format_node_values(transformed),
        block.end_marker,
    ]


def _rewrite_2414(block: RawDatasetBlock, mode: ModeShape, spec: TransformSpec) -> list[str]:
    transformed = transformed_mode_shape(mode, spec)
    header = block.content_lines[:13]
    return [
        block.start_marker,
        block.type_line,
        *header,
        *_format_node_values(transformed),
        block.end_marker,
    ]


def _format_node_values(values_by_node: dict[int, np.ndarray]) -> list[str]:
    lines: list[str] = []
    for node in sorted(values_by_node):
        lines.append(f"{node:10d}\n")
        lines.extend(_format_float_chunks(values_by_node[node], width=13, precision=5, per_line=6))
    return lines


def _format_floats(values: Iterable[float], width: int, precision: int, per_line: int) -> str:
    return "".join(f"{float(value):{width}.{precision}E}" for value in values) + "\n"


def _format_float_chunks(
    values: Iterable[float], width: int, precision: int, per_line: int
) -> list[str]:
    chunks: list[str] = []
    current: list[float] = []
    for value in values:
        current.append(float(value))
        if len(current) == per_line:
            chunks.append(_format_floats(current, width, precision, per_line))
            current = []
    if current:
        chunks.append(_format_floats(current, width, precision, per_line))
    return chunks


def _normal_mode_reals_from_2414(
    real_params: list[float],
) -> tuple[float | None, float | None, float | None, float | None]:
    if not real_params:
        return None, None, None, None
    if len(real_params) >= 6 and real_params[0] == 0.0 and real_params[1] != 0.0:
        return (
            real_params[1],
            real_params[3] if len(real_params) > 3 else None,
            real_params[4] if len(real_params) > 4 else None,
            real_params[5] if len(real_params) > 5 else None,
        )
    return (
        real_params[0],
        real_params[1] if len(real_params) > 1 else None,
        real_params[2] if len(real_params) > 2 else None,
        real_params[3] if len(real_params) > 3 else None,
    )


def _mode_number_from_2414(label: int | None, int_params: list[int]) -> int | None:
    for index in (5, 4):
        if len(int_params) > index and int_params[index] != 0:
            return int_params[index]
    return label


def _ints(line: str) -> list[int]:
    out: list[int] = []
    for part in line.split():
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _floats(line: str) -> list[float]:
    out: list[float] = []
    for part in line.replace("D", "E").replace("d", "E").split():
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "E"))


def _first_int(line: str) -> int | None:
    values = _ints(line)
    return values[0] if values else None


def _looks_like_node_label_line(line: str) -> bool:
    parts = line.split()
    if len(parts) != 1:
        return False
    token = parts[0]
    if token[:1] in {"+", "-"}:
        token = token[1:]
    return token.isdigit()


def _safe_int(parts: list[str], index: int) -> int | None:
    try:
        return int(parts[index])
    except (IndexError, ValueError):
        return None


def _safe_float(parts: list[str], index: int) -> float | None:
    try:
        return _float(parts[index])
    except (IndexError, ValueError):
        return None


def _line(text: str) -> str:
    return text if text.endswith(("\n", "\r")) else f"{text}\n"
