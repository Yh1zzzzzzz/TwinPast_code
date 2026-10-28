# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from .official import SILENT
from .compat import _use_decord_video_backend, _use_fast_patch_embed

def processor(model_path):
    from transformers import Qwen3VLProcessor
    from transformers.models.qwen3_vl import video_processing_qwen3_vl as vp
    _use_decord_video_backend()
    # Official README patches these two processor settings. Scope is this
    # isolated process, never the installed package or checkpoint files.
    if not getattr(vp.smart_resize, '_aura_patched', False):
        original = vp.smart_resize
        def smart_resize(num_frames, height, width, temporal_factor=2, **kwargs):
            return original(max(num_frames, temporal_factor), height, width,
                            temporal_factor=temporal_factor, **kwargs)
        smart_resize._aura_patched = True
        vp.smart_resize = smart_resize
    p = Qwen3VLProcessor.from_pretrained(str(model_path), local_files_only=True)
    p.video_processor.min_frames = 2
    return p

class TransformersBackend:
    def __init__(self, model_path, max_tokens=128, gpu_memory_gib=20):
        import torch
        import transformers
        from transformers import Qwen3VLForConditionalGeneration
        if transformers.__version__ != '4.57.1':
            raise RuntimeError('AURA requires Transformers 4.57.1')
        self.torch, self.max_tokens = torch, max_tokens
        self.processor = processor(model_path)
        if self.processor.tokenizer.convert_tokens_to_ids(SILENT) != 151669:
            raise ValueError('Checkpoint lacks the trained AURA silent token')
        _use_fast_patch_embed()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path),
            local_files_only=True, torch_dtype=torch.bfloat16, device_map='balanced',
            max_memory={i:f'{gpu_memory_gib}GiB' for i in range(torch.cuda.device_count())},
            attn_implementation='sdpa').eval()
        devices = sorted({str(p.device) for p in self.model.parameters()})
        if any(not x.startswith('cuda') for x in devices):
            raise RuntimeError(f'Insufficient GPU memory; unexpected offload to {devices}')
        self.metadata = {'backend':'transformers','transformers':transformers.__version__,
            'torch':torch.__version__, 'dtype':'bfloat16','devices':devices,
            'attention':'sdpa','fast_patch_embed':True,'model_path':str(Path(model_path).resolve()),
            'processor_min_frames':2, 'fps':1.0, 'pixel_cap':602112,
            'max_tokens':max_tokens, 'temperature':0, 'use_cache':True}
