# SPDX-License-Identifier: Apache-2.0
"""Processor decoding and equivalent vision patch projection compatibility."""
def _use_decord_video_backend() -> None:
    """Apply an equivalent inference compatibility implementation."""
    from transformers.video_processing_utils import BaseVideoProcessor
    from transformers.video_utils import load_video

    def fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
        if isinstance(video_url_or_urls, list):
            return list(zip(*[self.fetch_videos(x, sample_indices_fn=sample_indices_fn) for x in video_url_or_urls]))
        return load_video(video_url_or_urls, backend="decord", sample_indices_fn=sample_indices_fn)

    BaseVideoProcessor.fetch_videos = fetch_videos

def _use_fast_patch_embed() -> None:
    """Apply an equivalent inference compatibility implementation."""
    import torch.nn.functional as F

    def forward(self, hidden_states):
        weight = self.proj.weight
        if hidden_states.dim() == 5:
            hidden_states = hidden_states.reshape(-1, weight[0].numel())
        return F.linear(hidden_states.to(weight.dtype), weight.reshape(weight.shape[0], -1), self.proj.bias)

    for module, name in (("qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VisionPatchEmbed"),
                         ("qwen3_vl.modeling_qwen3_vl", "Qwen3VLVisionPatchEmbed"),
                         ("qwen3_5.modeling_qwen3_5", "Qwen3_5VisionPatchEmbed")):
        try:
            import importlib
            cls = getattr(importlib.import_module(f"transformers.models.{module}"), name)
            cls.forward = forward
        except Exception:
            pass

