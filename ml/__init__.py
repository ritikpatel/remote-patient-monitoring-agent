"""Phase 5 -- predictive models (PROJECT_PLAN.md section 11)."""

import os

# LightGBM and PyTorch each bundle their own OpenMP runtime; loading both in
# one macOS process is a documented crash (reproduced directly while building
# this package: a clean run of LightGBM CV fits, immediately followed by GRU
# training in the same process, SIGSEGV'd with no Python traceback -- exit
# 139). Set here, at package import time, so it applies whether ``ml.models``
# is used from ``ml/evaluation/run_all.py`` or from the pytest suite, and
# before any submodule has had a chance to import numpy/lightgbm/torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
