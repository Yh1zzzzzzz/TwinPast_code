# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Keep Perception/Decision together and transfer its outputs to Reaction."""
import types


def configure_sdpa():
    from dispider.model.decision_backbone.adapter import Qwen2Model
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as BaseQwen2Model
    # Released __init__ consists only of setting FA2 and calling this base
    # constructor. Use PyTorch SDPA for the same full causal attention graph.
    def init(self, config):
        config._attn_implementation = 'sdpa'
        BaseQwen2Model.__init__(self, config)
    Qwen2Model.__init__ = init


def load_distributed(native_loader, *args, **kwargs):
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Dispider evaluation requires two reserved GPUs')
    configure_sdpa()
    # The custom Decision decoder has internal tensor concatenations; keep it
    # intact. Reaction (7B) and its projectors fit on the other A5000.
    # Load on card 0 first (released FP16 weights total ~18.2GiB), then move
    # the whole custom component. Automatic dispatch hooks can incorrectly
    # reroute indices in Decision's direct forward_token method calls.
    kwargs['device_map'] = {'': 0}
    kwargs['max_memory'] = {0: '20GiB', 1: '20GiB'}
    kwargs['attn_implementation'] = 'sdpa'
    result = native_loader(*args, **kwargs)
    model = result[1]
    model.get_perception_decision().to('cuda:1')
    model.hf_device_map = {'model.embed_tokens':0,'model.layers':0,'model.norm':0,
        'lm_head':0,'model.compressor':1,'model.compress_projector':0,'model.clip_projector':0}
    install_boundary_transfer(model)
    return result


def install_boundary_transfer(model):
    from accelerate.utils import send_to_device
    component = model.get_perception_decision()
    original = component.forward
    source = component.device
    destination = model.get_input_embeddings().weight.device

    def forward(self, *args, **kwargs):
        values = original(*send_to_device(args, source),
                          **send_to_device(kwargs, source))
        # Projectors and text embeddings belong to Reaction. Transfers preserve
        # dtype and values; no compression, pooling or input reduction is added.
        return send_to_device(values, destination)

    component.forward = types.MethodType(forward, component)
