# Building With PyInstaller

This project supports a Windows PyInstaller `onedir` build. `onedir` is the recommended format for PyQt6 + VTK/PyVista applications because it is easier to inspect and more reliable than `onefile`.

## Prerequisites

- Windows
- `uv`
- Python 3.11, 3.12, or 3.13. Python 3.12 is the tested build target.

## Build

From the project root:

```powershell
cd C:\Users\emre_\PycharmProjects\unv_modal_viewer
.\scripts\build_pyinstaller.ps1
```

The script will:

1. Create `.venv` with Python 3.12 if needed.
2. Install the app in editable mode with `test` and `build` extras.
3. Run the regression tests.
4. Run PyInstaller using `unv_modal_viewer.spec`.
5. Write the frozen app to `dist\unv-modal-viewer\unv-modal-viewer.exe`.

To skip tests during local iteration:

```powershell
.\scripts\build_pyinstaller.ps1 -SkipTests
```

## Manual Build Commands

```powershell
uv venv --python 3.12
uv pip install -e ".[test,build]"
uv run pytest
uv run pyinstaller .\unv_modal_viewer.spec --clean --noconfirm
```

## Smoke Test

Check that the executable exists:

```powershell
.\scripts\smoke_frozen_import.ps1
```

Run the GUI manually:

```powershell
.\dist\unv-modal-viewer\unv-modal-viewer.exe path\to\modal_test_file.unv
```

Good manual files to try are the public modal-test UFF files used by the automated tests:

- `https://raw.githubusercontent.com/ladisk/pyuff/main/data/beam.uff`
- `https://raw.githubusercontent.com/ladisk/pyuff/main/data/2411%20and%202414.uff`
- `https://raw.githubusercontent.com/ladisk/pyuff/main/data/uff55_translation.uff`

## Notes

- The checked-in spec collects PyQt6, `qtpy`, PyVista, PyVistaQt, and VTK modules explicitly. This is intentionally broad because VTK uses many dynamic imports.
- Keep the build as `onedir` unless there is a strong distribution reason to switch. A `onefile` build will be larger, slower to start, and more sensitive to VTK plugin/resource extraction.
- If the GUI opens but the viewport is blank, verify graphics drivers/OpenGL support and test a normal Python run with `python -m unv_modal_viewer`.
