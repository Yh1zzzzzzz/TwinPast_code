# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Same proactive controller/input; explicitly enable per-generation KV reuse.

No cross-request cache, shortened history, altered sampling or new checkpoint.
Kept separate because cached and uncached GPU arithmetic need not be identical.
"""
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
from streaming_proactive_adapter import Adapter as ProactiveAdapter
from common import ROOT


class Adapter(ProactiveAdapter):
    def __init__(self):
        super().__init__()
        self.main.use_cache=True
        self.summary.use_cache=True
        self.root=RUN_ROOT / 'joyai_frames'
        self.controller.config.frame_save_dir=str(self.root)
        self.controller.config.allowed_local_image_roots=(str(self.root),)

    async def _answer_stream(self,job,prompt,emit):
        result=await super()._answer_stream(job,prompt,emit)
        result.update(transport='local HF with explicit per-generation use_cache=True; no cross-request prefix cache',
                      readout_protocol='free_generation',condition='joyai_streaming',
                      main_use_cache=True,summary_use_cache=True)
        return result
