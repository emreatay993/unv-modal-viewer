from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import unv_modal_viewer.app


def test_main_script_runs_without_package_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """PyInstaller executes __main__.py as a script, not as package-relative code."""
    monkeypatch.setattr(unv_modal_viewer.app, "main", lambda: 0)
    entrypoint = Path(__file__).parents[1] / "src" / "unv_modal_viewer" / "__main__.py"

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(entrypoint), run_name="__main__")

    assert exc_info.value.code == 0
