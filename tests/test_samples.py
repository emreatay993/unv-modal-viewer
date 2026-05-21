from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pytest

from unv_modal_viewer.io import load_unv
from unv_modal_viewer.samples import SAMPLES


def test_artemis_geometry_sample_is_available() -> None:
    url = SAMPLES["Artemis geometry.uff"]

    assert "Artemis%20export%20-%20Geometry" in url
    assert url.endswith("RPBC_setup_05_14102016_105117.uff")


def test_public_pyuff_artemis_geometry_fixture_loads(tmp_path: Path) -> None:
    url = SAMPLES["Artemis geometry.uff"]
    path = tmp_path / "artemis_geometry.uff"
    try:
        urlretrieve(url, path)
    except Exception as exc:  # pragma: no cover - network dependent skip
        pytest.skip(f"public Artemis geometry fixture unavailable: {exc}")

    model = load_unv(path)

    assert model.metadata["dataset_counts"][15] == 1
    assert model.metadata["dataset_counts"][82] == 2
    assert model.metadata["dataset_counts"][2412] == 1
    assert len(model.nodes) == 74
    assert len(model.elements) == 108
