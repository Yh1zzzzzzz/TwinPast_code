# SPDX-License-Identifier: Apache-2.0
import json, os
from pathlib import Path
MODEL_ROOT = Path(os.environ.get('TWINPAST_MODEL_ROOT', Path.cwd() / 'models'))
RUN_ROOT = Path(os.environ.get('TWINPAST_RUN_ROOT', Path.cwd() / 'runs'))
CONDITIONS = ('tail30', 'streaming')
def stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, inode=value.st_ino, mtime_ns=value.st_mtime_ns)
def check(job):
    if stat(job['video']) != job['video_stat']:
        raise ValueError('Video changed after input preparation')
def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default
def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
