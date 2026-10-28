"""Inspect or install registered model sources without changing existing checkouts."""
import json
from pathlib import Path
import subprocess

from .data import DataError, require

PACKAGE = Path(__file__).resolve().parent


def sources():
    return json.loads((PACKAGE / "upstreams.json").read_text())


def setup(model=None, fetch=False):
    registered = sources()
    require(model is None or model in registered, "Unknown upstream model")
    results = []
    for name, source in registered.items():
        if model and name != model:
            continue
        path = PACKAGE / "adapters" / name / source.get("directory", "upstream")
        if path.exists():
            existing = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True)
            results.append({"model": name, "status": "present" if existing.returncode == 0 and existing.stdout.strip() == source["revision"] else "revision_differs"})
        elif fetch:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(["git", "clone", "--no-checkout", source["repository"], str(path)], check=True)
                subprocess.run(["git", "-C", str(path), "checkout", "--detach", source["revision"]], check=True)
            except subprocess.CalledProcessError:
                raise DataError("Upstream checkout failed for " + name) from None
            results.append({"model": name, "status": "installed"})
        else:
            results.append({"model": name, "status": "absent", **source})
    return results

