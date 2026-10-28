# SPDX-License-Identifier: Apache-2.0
"""Offline sparse-codec inference using the Mage-VL checkpoint interface."""
import os

from common import CACHE_ROOT, MODEL_ROOT, PIXELS, load_kwargs, record_inputs


class Adapter:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        os.environ['ONLINE_CODEC_CACHE_DIR'] = str(CACHE_ROOT / 'artifacts/codec_cache')
        path = str(MODEL_ROOT / 'Mage-VL')
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs()).eval()
        self.processor.video_processor.max_pixels = PIXELS
        self.processor.video_processor.image_processor.max_pixels = PIXELS

    def answer(self, frames, meta, prompt, video):
        import torch
        p = self.processor
        text = p.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'video'}, {'type': 'text', 'text': prompt}]}],
            tokenize=False, add_generation_prompt=True)
        assert meta['indices'][-1] == meta['source_frames'] - 1, 'Codec source has post-query frames; causal trim required'
        inputs = p(text=[text], videos=[video], video_backend='codec', max_pixels=PIXELS,
                   codec_config={'engine': 'hevc', 'target_canvas': 64, 'patch': 16},
                   return_tensors='pt', padding=True)
        info = record_inputs(inputs)
        positions = inputs['patch_positions'].detach().cpu()
        source_ids = torch.unique(positions[positions[:, 0] >= 0, 0]).tolist()
        info['codec_input_sampling'] = {
            'engine': 'hevc', 'target_canvas': 64,
            'requested_raw_samples': min(512, meta['source_frames']),
            'represented_source_frame_indices': source_ids,
            'patch_positions_shape': list(positions.shape),
            'cache_root': os.environ['ONLINE_CODEC_CACHE_DIR']}
        device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(device, dtype=self.model.dtype) if k == 'pixel_values'
                  else v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        ids = output[0, inputs['input_ids'].shape[-1]:]
        return dict(response=p.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                    output_tokens=len(ids), backend='codec', **info)
