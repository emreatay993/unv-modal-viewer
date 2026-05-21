from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from qtpy.QtCore import QStandardPaths


SAMPLES: dict[str, str] = {
    "beam.uff": "https://raw.githubusercontent.com/ladisk/pyuff/main/data/beam.uff",
    "2411 and 2414.uff": "https://raw.githubusercontent.com/ladisk/pyuff/main/data/2411%20and%202414.uff",
    "uff55_translation.uff": "https://raw.githubusercontent.com/ladisk/pyuff/main/data/uff55_translation.uff",
    "Artemis geometry.uff": (
        "https://raw.githubusercontent.com/ladisk/pyuff/main/data/"
        "Artemis%20export%20-%20Geometry%20RPBC_setup_05_14102016_105117.uff"
    ),
}


def sample_cache_dir() -> Path:
    root = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    if not root:
        root = str(Path.home() / ".unv_modal_viewer")
    path = Path(root) / "samples"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_sample_path(name: str) -> Path:
    return sample_cache_dir() / name


def ensure_sample_file(name: str) -> Path:
    if name not in SAMPLES:
        raise KeyError(f"Unknown sample: {name}")
    path = cached_sample_path(name)
    if path.exists() and path.stat().st_size > 0:
        return path
    urlretrieve(SAMPLES[name], path)
    return path
