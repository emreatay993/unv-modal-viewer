from __future__ import annotations

from pathlib import Path

from unv_modal_viewer.settings import AppSettings
from unv_modal_viewer.state import RenderOptions


def _qsettings(path: Path):
    from qtpy.QtCore import QSettings

    ini_format = getattr(QSettings, "IniFormat", None) or QSettings.Format.IniFormat
    return QSettings(str(path), ini_format)


def test_recent_files_are_deduplicated_and_ordered(tmp_path: Path) -> None:
    settings = AppSettings(_qsettings(tmp_path / "settings.ini"))
    first = tmp_path / "first.unv"
    second = tmp_path / "second.unv"

    settings.add_recent_file(first)
    settings.add_recent_file(second)
    settings.add_recent_file(first)

    assert settings.recent_files() == [first.resolve(), second.resolve()]


def test_render_options_round_trip(tmp_path: Path) -> None:
    settings = AppSettings(_qsettings(tmp_path / "settings.ini"))
    options = RenderOptions(colormap="plasma", reverse_colormap=True, legend_position="Right", point_size=14)

    settings.save_render_options(options)
    restored = AppSettings(_qsettings(tmp_path / "settings.ini")).load_render_options()

    assert restored.colormap == "plasma"
    assert restored.reverse_colormap is True
    assert restored.legend_position == "Right"
    assert restored.point_size == 14


def test_view_and_overlay_preferences_round_trip(tmp_path: Path) -> None:
    settings = AppSettings(_qsettings(tmp_path / "settings.ini"))

    settings.save_view_flags(points=False, surface=True, generated_surface=False, traces=True)
    settings.save_overlay_preferences(opacity=0.44, color="#22c55e")
    restored = AppSettings(_qsettings(tmp_path / "settings.ini"))

    assert restored.load_view_flags() == {
        "points": False,
        "surface": True,
        "generated_surface": False,
        "traces": True,
    }
    assert restored.load_overlay_preferences() == {"opacity": 0.44, "color": "#22c55e"}
