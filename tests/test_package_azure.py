import importlib.util
import zipfile

import pytest

from routesense.config import PROJECT_DIR

_spec = importlib.util.spec_from_file_location("package_azure", PROJECT_DIR / "scripts" / "package_azure.py")
package = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(package)


def _fake_model(processed, with_eval=True):
    region = processed / package.REGION
    (region / "artifacts").mkdir(parents=True)
    (region / "artifacts" / "meta.json").write_text("{}")
    (region / "artifacts" / "risk_table.npz").write_bytes(b"npz")
    (region / "segments.parquet").write_bytes(b"parquet")
    (region / "sim").mkdir()
    (region / "sim" / "events.parquet").write_bytes(b"not needed at runtime")
    if with_eval:
        (region / "eval").mkdir()
        (region / "eval" / "report.json").write_text("{}")


def test_zip_has_code_model_and_pinned_requirements(tmp_path):
    _fake_model(tmp_path / "processed")
    out = tmp_path / "app.zip"
    names = package.build_zip(out, tmp_path / "processed")
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == sorted(names)
        reqs = z.read("requirements.txt").decode()
    for name in ("routesense/api/app.py", "web/index.html", "data/processed/blr_core/artifacts/meta.json",
                 "data/processed/blr_core/segments.parquet", "data/processed/blr_core/eval/report.json"):
        assert name in names
    assert not any("__pycache__" in n or "\\" in n or n.startswith(("tests/", "data/processed/blr_core/sim/"))
                   for n in names)
    pins = dict(line.split("==") for line in reqs.split())
    assert set(pins) == set(package.PACKAGES) and all(pins.values())


def test_eval_report_is_optional(tmp_path):
    _fake_model(tmp_path / "processed", with_eval=False)
    names = package.build_zip(tmp_path / "app.zip", tmp_path / "processed")
    assert "data/processed/blr_core/artifacts/meta.json" in names


def test_needs_a_built_model(tmp_path):
    with pytest.raises(SystemExit):
        package.build_zip(tmp_path / "app.zip", tmp_path / "nothing-built")
