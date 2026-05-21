from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QSettings

from .state import RenderOptions


class AppSettings:
    def __init__(self, settings: QSettings | None = None) -> None:
        self.settings = settings or QSettings()

    def recent_files(self) -> list[Path]:
        values = self.settings.value("recent/files", [], type=list)
        return [Path(str(value)) for value in values if str(value)]

    def add_recent_file(self, path: str | Path, limit: int = 10) -> None:
        resolved = str(Path(path).resolve())
        current = [str(value) for value in self.recent_files()]
        current = [value for value in current if value != resolved]
        self.settings.setValue("recent/files", [resolved, *current][:limit])

    def last_folder(self) -> Path:
        value = self.settings.value("recent/last_folder", str(Path.home()))
        return Path(str(value))

    def set_last_folder(self, path: str | Path) -> None:
        folder = Path(path)
        self.settings.setValue("recent/last_folder", str(folder if folder.is_dir() else folder.parent))

    def save_window(self, window: object, splitter: object, sections: dict[str, object]) -> None:
        self.settings.setValue("window/geometry", window.saveGeometry())
        self.settings.setValue("window/splitter", splitter.saveState())
        for name, section in sections.items():
            self.settings.setValue(f"sections/{name}", section.is_expanded())

    def restore_window(self, window: object, splitter: object, sections: dict[str, object]) -> None:
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            window.restoreGeometry(geometry)
        splitter_state = self.settings.value("window/splitter")
        if splitter_state is not None:
            splitter.restoreState(splitter_state)
        for name, section in sections.items():
            value = self.settings.value(f"sections/{name}", None)
            if value is not None:
                section.set_expanded(_bool(value))

    def save_render_options(self, options: RenderOptions) -> None:
        self.settings.setValue("render/colormap", options.colormap)
        self.settings.setValue("render/reverse_colormap", options.reverse_colormap)
        self.settings.setValue("render/scalar_auto", options.scalar_auto)
        self.settings.setValue("render/scalar_min", options.scalar_min)
        self.settings.setValue("render/scalar_max", options.scalar_max)
        self.settings.setValue("render/legend_visible", options.legend_visible)
        self.settings.setValue("render/legend_position", options.legend_position)
        self.settings.setValue("render/surface_opacity", options.surface_opacity)
        self.settings.setValue("render/point_size", options.point_size)
        self.settings.setValue("render/selected_color", options.selected_color)

    def load_render_options(self) -> RenderOptions:
        defaults = RenderOptions()
        return RenderOptions(
            colormap=str(self.settings.value("render/colormap", defaults.colormap)),
            reverse_colormap=_bool(self.settings.value("render/reverse_colormap", defaults.reverse_colormap)),
            scalar_auto=_bool(self.settings.value("render/scalar_auto", defaults.scalar_auto)),
            scalar_min=float(self.settings.value("render/scalar_min", defaults.scalar_min)),
            scalar_max=float(self.settings.value("render/scalar_max", defaults.scalar_max)),
            legend_visible=_bool(self.settings.value("render/legend_visible", defaults.legend_visible)),
            legend_position=str(self.settings.value("render/legend_position", defaults.legend_position)),
            surface_opacity=float(self.settings.value("render/surface_opacity", defaults.surface_opacity)),
            point_size=int(self.settings.value("render/point_size", defaults.point_size)),
            selected_color=str(self.settings.value("render/selected_color", defaults.selected_color)),
        )

    def save_view_flags(self, points: bool, surface: bool, generated_surface: bool, traces: bool) -> None:
        self.settings.setValue("view/points", points)
        self.settings.setValue("view/surface", surface)
        self.settings.setValue("view/generated_surface", generated_surface)
        self.settings.setValue("view/traces", traces)

    def load_view_flags(self) -> dict[str, bool]:
        return {
            "points": _bool(self.settings.value("view/points", True)),
            "surface": _bool(self.settings.value("view/surface", True)),
            "generated_surface": _bool(self.settings.value("view/generated_surface", True)),
            "traces": _bool(self.settings.value("view/traces", True)),
        }

    def save_overlay_preferences(self, opacity: float, color: str) -> None:
        self.settings.setValue("overlay/opacity", float(opacity))
        self.settings.setValue("overlay/color", color)

    def load_overlay_preferences(self) -> dict[str, object]:
        return {
            "opacity": float(self.settings.value("overlay/opacity", 0.32)),
            "color": str(self.settings.value("overlay/color", "#f59e0b")),
        }


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)
