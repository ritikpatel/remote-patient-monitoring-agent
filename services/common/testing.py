"""Shared test helper: load a service's module by file path under a unique name.

Every service's main file is named `app.py` (a deliberate, uniform convention --
see each Dockerfile's `--app-dir` CMD). That is fine for running one service at a
time, but running the WHOLE repo's test suite in one pytest process means many
`tests/test_app.py` files each do the equivalent of `import app`: the first one
populates `sys.modules["app"]`, and every subsequent service's test file silently
gets THAT cached module back instead of its own -- found by running the full suite
(`pytest` with no path filter) after every individual service's tests passed in
isolation, which is exactly the discrepancy that exposed it.

`load_module` sidesteps this by loading each file under a name derived from its own
path, so `services/risk-engine/app.py` and `services/rag-service/app.py` never
collide in `sys.modules` no matter what order pytest collects them in.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module(path: Path, unique_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    # The loaded module's own top-of-file `sys.path.insert(0, str(Path(__file__)...))`
    # calls (every service's app.py has these, to reach services/common etc.) need
    # its own directory on sys.path too, for any bare `import <local_module>` inside it.
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def load_service_app(service_dir_name: str, repo_root: Path) -> ModuleType:
    """service_dir_name e.g. "risk-engine" -> loads services/risk-engine/app.py
    under the unique module name "svc_risk_engine_app"."""
    unique_name = "svc_" + service_dir_name.replace("-", "_") + "_app"
    return load_module(repo_root / "services" / service_dir_name / "app.py", unique_name)
