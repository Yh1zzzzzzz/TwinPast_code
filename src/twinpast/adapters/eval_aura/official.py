# SPDX-License-Identifier: Apache-2.0
"""Load registered upstream history definitions without server startup side effects."""
import ast, base64, contextlib, io, json, math, os, re, subprocess, time
from datetime import datetime
from pathlib import Path
VENDOR = Path(__file__).resolve().parent / 'upstream'
SILENT = '<|silent|>'

def load_definitions(path, names):
    """Execute only requested, reviewed function/class ASTs; no upstream top-level code."""
    parsed = ast.parse(Path(path).read_text())
    found = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))
             and (getattr(node, 'name', None) in names or isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id in names for target in node.targets))]
    if len(found) != len(names):
        raise ValueError(f'Missing upstream definitions in {path}: {names}')
    namespace = {'base64': base64, 'math': math, 'os': os, 'subprocess': subprocess,
                 'time': time, 'datetime': datetime, 'json': json, 're': re, 'SILENT_TEXT': SILENT}
    exec(compile(ast.Module(body=found, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def upstream_bench():
    path = VENDOR / 'AURA_bench_eval/OVO-Bench/models/AURA.py'
    return load_definitions(path, {'SYSTEM_PROMPT', 'encode_video_base64', 'get_video_duration', 'split_video_from_end'})

class StreamingHistory:
    """Thin wrapper around upstream SessionHistory, N=30, margin=15, M=10.

    Observations must be actual model output; no supplied ground-truth history.
    Prefix KV reuse is a backend responsibility, not claimed by this wrapper.
    """
    def __init__(self, video_rounds=30, margin=15, qa_groups=10, system_prompt=None):
        if video_rounds < 1 or margin < 0 or qa_groups < 1:
            raise ValueError('Invalid history window')
        cls = load_definitions(VENDOR / 'Qwen3_VL_online_streaming_v2_ContextManaged.py', {'SessionHistory'})['SessionHistory']
        with contextlib.redirect_stdout(io.StringIO()):
            self.delegate = cls(max_rounds=video_rounds + margin, num_rounds_keep=video_rounds,
                                max_context_qas=qa_groups, pruning_enabled=True)
        if system_prompt is not None:
            self.delegate.system_prompt = system_prompt
            self.delegate._system_msg = {'role': 'system', 'content': system_prompt}
            self.delegate._rebuild_history()

    def add_chunk(self, video, text=''):
        with contextlib.redirect_stdout(io.StringIO()):
            self.delegate.add_user_message(text, video_tuple=video)

    def add_response(self, response):
        self.delegate.add_assistant_message(response)

    @property
    def messages(self):
        return self.delegate.history

    @property
    def state(self):
        return {'video_rounds': self.delegate._sw_round_count(),
                'text_qa_groups': len(self.delegate._context_history)}

