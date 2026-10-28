# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Causal frame replay for native incremental APIs; no model-specific memory policy."""
import json
import math
from pathlib import Path
import numpy as np


class CausalFrames:
    def __init__(self, job, fps, include_query_frame=False):
        from decord import VideoReader, cpu
        self.path = Path(job['video'])
        st = self.path.stat()
        assert dict(size=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns) == job['video_stat']
        self.reader = VideoReader(str(self.path), ctx=cpu(0), num_threads=1)
        self.source_fps = float(self.reader.get_avg_fps())
        pts = np.asarray(self.reader.get_frame_timestamp(np.arange(len(self.reader))))[:, 0]
        # The video collection uses constant 24 FPS. Use exact frame-index
        # times to avoid decord float32 timestamp rounding at the query boundary.
        exact = np.arange(len(self.reader)) / self.source_fps
        assert abs(self.source_fps - 24) < 1e-6
        assert np.max(np.abs(pts - exact)) < 1e-3
        targets = np.arange(math.floor(job['t_query'] * fps + 1e-8) + 1) / fps
        self.indices = np.unique(np.floor(targets * self.source_fps + 1e-8).astype(int))
        self.indices = self.indices[(self.indices < len(self.reader)) & (self.indices / self.source_fps <= job['t_query'] + 1e-9)]
        if include_query_frame:
            last=min(len(self.reader)-1, math.floor(job['t_query']*self.source_fps+1e-8))
            self.indices=np.unique(np.append(self.indices,last))
        self.meta = dict(source=str(self.path), source_fps=self.source_fps, replay_fps=fps,
                         indices=self.indices.tolist(), timestamps_s=(self.indices / self.source_fps).tolist(),
                         raw_decord_pts_s=pts[self.indices].tolist(), t_query=job['t_query'],
                         sampling='causal_fixed_cadence_no_global_frame_cap', common_resize=False,
                         include_query_frame=include_query_frame)

    def __iter__(self):
        from PIL import Image
        for index in self.indices:
            yield Image.fromarray(self.reader[int(index)].asnumpy()), int(index) / self.source_fps
