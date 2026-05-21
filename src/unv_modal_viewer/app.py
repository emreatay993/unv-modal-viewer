from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("QT_API", "pyqt6")

    args = list(sys.argv[1:] if argv is None else argv)
    from qtpy.QtWidgets import QApplication
    from .gui import MainWindow, set_fusion_theme

    app = QApplication.instance() or QApplication(sys.argv[:1] + args)
    set_fusion_theme(app)
    initial = Path(args[0]) if args else None
    window = MainWindow(initial)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
