# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


root = Path.cwd()
src_root = root / "src"

hiddenimports = [
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.sip",
    "qtpy",
    "qtpy.QtCore",
    "qtpy.QtGui",
    "qtpy.QtWidgets",
    "pyuff",
    "pyvista",
    "pyvista.plotting",
    "pyvistaqt",
    "pyvistaqt.plotting",
    "vtk",
    "vtkmodules.vtkCommonCore",
    "vtkmodules.vtkCommonDataModel",
    "vtkmodules.vtkFiltersCore",
    "vtkmodules.vtkFiltersGeneral",
    "vtkmodules.vtkFiltersSources",
    "vtkmodules.vtkInteractionStyle",
    "vtkmodules.vtkInteractionWidgets",
    "vtkmodules.vtkRenderingAnnotation",
    "vtkmodules.vtkRenderingCore",
    "vtkmodules.vtkRenderingFreeType",
    "vtkmodules.vtkRenderingOpenGL2",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
]

for package in ("vtkmodules",):
    hiddenimports += collect_submodules(package)

hiddenimports = [
    name
    for name in hiddenimports
    if not (
        name.startswith("qtpy.tests")
        or name.startswith("pyvista.tests")
        or name.startswith("vtkmodules.test")
        or name.startswith("vtkmodules.tk")
        or name.startswith("vtkmodules.wx")
        or name.startswith("vtkmodules.gtk")
    )
]

datas = []
for package in ("pyvista", "pyvistaqt", "vtkmodules"):
    datas += collect_data_files(package)

binaries = collect_dynamic_libs("vtkmodules")


a = Analysis(
    [str(src_root / "unv_modal_viewer" / "__main__.py")],
    pathex=[str(src_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "tests",
        "IPython",
        "jupyter",
        "notebook",
        "tkinter",
        "qtpy.tests",
        "pyvista.tests",
        "vtkmodules.test",
        "vtkmodules.tk",
        "vtkmodules.wx",
        "vtkmodules.gtk",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="unv-modal-viewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="unv-modal-viewer",
)
