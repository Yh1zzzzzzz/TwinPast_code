# SPDX-License-Identifier: Apache-2.0
"""User-configured model and output locations."""
import os
from pathlib import Path
ROOT = Path.cwd()
ADAPTERS = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get('TWINPAST_MODEL_ROOT', ROOT / 'models')).expanduser().resolve()
RUN_ROOT = Path(os.environ.get('TWINPAST_RUN_ROOT', ROOT / 'runs')).expanduser().resolve()
