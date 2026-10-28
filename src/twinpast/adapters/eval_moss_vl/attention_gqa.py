# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Full-context grouped cross attention without materializing repeated KV heads.

Neutral masks use native Flash SDPA GQA. Other masks use bounded blocks and
online softmax over ALL keys, preserving the mask, scaling and full context.
"""
import math
import torch
STATS={}
def reset_stats():
    STATS.clear();STATS.update(flash_gqa_calls=0,chunked_calls=0,max_key_tokens=0,max_query_tokens=0)
reset_stats()

def block_gqa(query,key,value,mask=None,scale=None,q_block=16,k_block=8192):
    batch,heads,nq,dim=query.shape;kvheads=key.shape[1];groups=heads//kvheads
    assert heads%kvheads==0 and key.shape==value.shape
    scale=scale if scale is not None else dim**-.5
    chunks=[]
    # Online log-sum-exp; no crop, sampling or retained-window change.
    with torch.autocast(device_type=query.device.type,enabled=False):
        for qs in range(0,nq,q_block):
            qe=min(qs+q_block,nq);q=query[:,:,qs:qe].float().reshape(batch,kvheads,groups,qe-qs,dim)
            maximum=torch.full((*q.shape[:-1],1),-float('inf'),device=q.device)
            denominator=torch.zeros_like(maximum);numerator=torch.zeros_like(q)
            for ks in range(0,key.shape[2],k_block):
                ke=min(ks+k_block,key.shape[2]);k=key[:,:,ks:ke].float();v=value[:,:,ks:ke].float()
                scores=torch.einsum('bhgqd,bhkd->bhgqk',q,k)*scale
                if mask is not None:
                    selected=mask[...,qs:qe,ks:ke] if mask.shape[-2]!=1 else mask[...,:,ks:ke]
                    selected=selected.expand(batch,heads,qe-qs,ke-ks).reshape(batch,kvheads,groups,qe-qs,ke-ks)
                    if selected.dtype==torch.bool:scores.masked_fill_(~selected,-float('inf'))
                    else:scores=scores+selected.float()
                new_maximum=torch.maximum(maximum,scores.amax(-1,keepdim=True))
                stable=torch.where(torch.isfinite(new_maximum),new_maximum,torch.zeros_like(new_maximum))
                old_weight=torch.exp(maximum-stable);weights=torch.exp(scores-stable)
                numerator=numerator*old_weight+torch.einsum('bhgqk,bhkd->bhgqd',weights,v)
                denominator=denominator*old_weight+weights.sum(-1,keepdim=True)
                maximum=new_maximum
            chunks.append((numerator/denominator.clamp_min(torch.finfo(torch.float32).tiny)).reshape(batch,heads,qe-qs,dim).to(query.dtype))
    return torch.cat(chunks,dim=2)

def cross_attention(query,key,value,mask=None,scale=None):
    STATS['max_key_tokens']=max(STATS['max_key_tokens'],key.shape[2]);STATS['max_query_tokens']=max(STATS['max_query_tokens'],query.shape[2])
    if mask is not None:mask=mask[...,:key.shape[2]]
    neutral=mask is None or bool((mask.all() if mask.dtype==torch.bool else (mask==0).all()).item())
    if neutral and query.device.type=='cuda':
        from torch.nn.attention import sdpa_kernel,SDPBackend
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output=torch.nn.functional.scaled_dot_product_attention(query,key,value,attn_mask=None,
                    dropout_p=0.,scale=scale,is_causal=False,enable_gqa=True)
            STATS['flash_gqa_calls']+=1;return output
        except RuntimeError as error:
            if not any(x in str(error).lower() for x in ['no available kernel','no viable backend']):raise
    STATS['chunked_calls']+=1
    return block_gqa(query,key,value,mask,scale)

def install():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    if getattr(original,'_moss_full_context_gqa',False):return
    def dispatch(module,query,key,value,attention_mask,dropout=0.,scaling=None,is_causal=None,**kwargs):
        if module.__class__.__name__!='MossVLTextCrossAttention':
            return original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,is_causal=is_causal,**kwargs)
        if dropout:raise RuntimeError('Inference-only full-context cross-attention requires dropout0')
        if is_causal:raise RuntimeError('MOSS cross-attention must remain noncausal; native mask handles visibility')
        out=cross_attention(query,key,value,attention_mask,scaling)
        return out.transpose(1,2).contiguous(),None
    dispatch._moss_full_context_gqa=True
    ALL_ATTENTION_FUNCTIONS.register('sdpa',dispatch)
