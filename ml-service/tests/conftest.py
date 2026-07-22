"""Pytest fixtures / environment for the ml-service test suite.

The backend refuses to start without a validated model (helmet_v4_clean). CI and
local test runs won't have those trained weights, so we explicitly opt into the
COCO-pretrained fallback for tests ONLY. This must be set before app.inference is
imported (it reads the flag at module load), which pytest guarantees by importing
conftest.py before the test modules.

This does NOT weaken production behavior: without this flag, `uvicorn app.main:app`
still refuses to start when the validated model is missing.
"""

import os

os.environ.setdefault("ALLOW_UNVALIDATED_MODEL", "1")
