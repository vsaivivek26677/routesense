"""Zip the web app with the locally built model for Azure App Service (Linux, Python).

    python scripts/run_region.py --region blr_core     # once, if data/processed/blr_core/ has no model yet
    python scripts/package_azure.py                    # -> dist/routesense-azure.zip

Azure installs requirements.txt from the zip. It is pinned to the package versions of the Python
environment running this script, so the cloud loads the model with the same libraries that built it.
Deployment steps: README.md, "Deploy a public demo".
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from importlib import metadata
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REGION = "blr_core"
CODE = ["routesense", "web"]
RUNTIME = ["segments.parquet", "artifacts", "eval"]  # what the API loads (eval is optional)
PACKAGES = ["osmnx", "geopandas", "shapely", "pyproj", "networkx", "numpy", "pandas", "pyarrow",
            "scikit-learn", "scipy", "joblib", "requests", "fastapi", "uvicorn"]


def pinned_requirements() -> str:
    """requirements.txt pinned to this environment's versions, else to requirements-deploy.txt."""
    fallback = {}
    for line in (PROJECT_DIR / "requirements-deploy.txt").read_text().splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.split("==")
            fallback[name.strip()] = version.strip()
    pins = []
    for pkg in PACKAGES:
        try:
            version = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            version = fallback[pkg]
        pins.append(f"{pkg}=={version}")
    return "\n".join(pins) + "\n"


def _files(root: Path) -> list[Path]:
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    return [p for p in paths if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]


def build_zip(out: Path, processed_dir: Path = PROCESSED_DIR) -> list[str]:
    """Write the deployment zip; return its entry names."""
    region_dir = processed_dir / REGION
    if not (region_dir / "artifacts" / "meta.json").exists():
        raise SystemExit(f"no built model in {region_dir}\nbuild it first: python scripts/run_region.py --region {REGION}")
    entries = []
    for name in CODE:
        entries += [(p, p.relative_to(PROJECT_DIR).as_posix()) for p in _files(PROJECT_DIR / name)]
    for name in RUNTIME:
        if (region_dir / name).exists():
            entries += [(p, f"data/processed/{REGION}/{p.relative_to(region_dir).as_posix()}")
                        for p in _files(region_dir / name)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path, arcname in entries:
            z.write(path, arcname)
        z.writestr("requirements.txt", pinned_requirements())
    return [arcname for _, arcname in entries] + ["requirements.txt"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=PROJECT_DIR / "dist" / "routesense-azure.zip")
    args = ap.parse_args()
    names = build_zip(args.out)
    if not any(n.startswith(f"data/processed/{REGION}/eval/") for n in names):
        print("note: no evaluation report found; the Evaluation tab will be empty")
    print(f"wrote {args.out} ({len(names)} files, {args.out.stat().st_size / 1e6:.0f} MB)")
    print(f"use the Azure runtime PYTHON:{sys.version_info.major}.{sys.version_info.minor} (this Python's version)")


if __name__ == "__main__":
    main()
