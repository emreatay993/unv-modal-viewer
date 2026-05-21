# UNV Modal Viewer

Standalone PyQt/PyVista viewer for ASCII UNV/UFF modal-test files.

## Features

- Reads common modal-test datasets: `15`, `55`, `58`, `82`, `164`, `2411`, `2412`, `2414`, and `2420`.
- Treats dataset `164` as units metadata.
- Preserves unknown or unsupported datasets during export.
- Visualizes test points, trace lines, file topology, generated triangulated surfaces, and dataset `55` / `2414` mode shapes.
- Supports coordinate scale, translation, and X/Y/Z degree rotation for coordinate-system alignment.
- Exports modified UNV/UFF coordinates, with an export toggle to also transform mode-shape vectors.
- Uses PyVista/PyVistaQt with a left vertical scalar legend and hover readout in the top-left viewport.
- Persists recent files, layout, view options, dark/light theme, legend/color options, and overlay preferences.
- Supports point selection, selected-node tables, hide/isolate selection, diagnostics, sample-file loading, overlay models, MAC/cross-MAC, and CSV/VTK/screenshot/MP4/AVI/GIF exports.

## Install

```powershell
cd C:\Users\emre_\PycharmProjects\unv_modal_viewer
uv venv --python 3.12
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

## Run

```powershell
.\.venv\Scripts\python.exe -m unv_modal_viewer path\to\file.unv
```

or:

```powershell
.\.venv\Scripts\unv-modal-viewer.exe path\to\file.unv
```

The public `pyuff` `beam.uff` file is used by tests as a real modal-test UFF fixture. It contains dataset `164`, `2420`, `2411`, and three dataset `58` FRFs. The app also exposes the public pyuff Artemis geometry export as a sample file. The tests generate compact dataset `55`/`2414` fixtures for true mode-shape and damping coverage.

## Build A Windows Executable

```powershell
.\scripts\build_pyinstaller.ps1
```

The frozen app is written to:

```powershell
.\dist\unv-modal-viewer\unv-modal-viewer.exe
```

See [docs/BUILDING.md](docs/BUILDING.md) for the full PyInstaller guide.
