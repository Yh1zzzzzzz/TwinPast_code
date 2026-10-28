# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
from common import *

def segmented_vision_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
 import torch,math
 from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision
 n=hidden_states.shape[0]
 q,k,v=self.qkv(hidden_states).reshape(n,3,self.num_heads,-1).permute(1,0,2,3).unbind(0)
 q=apply_rotary_pos_emb_vision(q.unsqueeze(0),rotary_pos_emb).squeeze(0)
 k=apply_rotary_pos_emb_vision(k.unsqueeze(0),rotary_pos_emb).squeeze(0)
 bounds=cu_seqlens.detach().cpu().tolist();groups={}
 for a,b in zip(bounds[:-1],bounds[1:]):groups.setdefault(b-a,[]).append((a,b))
 output=torch.empty_like(q)
 for spans in groups.values():
  qs=torch.stack([q[a:b] for a,b in spans]).transpose(1,2)
  ks=torch.stack([k[a:b] for a,b in spans]).transpose(1,2)
  vs=torch.stack([v[a:b] for a,b in spans]).transpose(1,2)
  values=torch.nn.functional.scaled_dot_product_attention(qs,ks,vs,dropout_p=0.0,is_causal=False).transpose(1,2)
  for value,(a,b) in zip(values,spans):output[a:b]=value
 return self.proj(output.reshape(n,-1))

class Adapter:
 def __init__(self):
  import sys
  up=ADAPTERS / 'eval_flash_vstream/upstream/Flash-VStream-Qwen'
  sys.path.insert(0,str(up))
  from models import FlashVStreamQwen2VLModel,FlashVStreamQwen2VLProcessor,DEFAULT_FLASH_MEMORY_CONFIG
  p=str(MODEL_ROOT / 'Flash-VStream-Qwen-7b')
  kw=load_kwargs(); kw['attn_implementation']='sdpa'
  self.model=FlashVStreamQwen2VLModel.from_pretrained(p,**kw).eval()
  import torch,types
  for vision_block in self.model.visual.blocks:
   attn=vision_block.attn
   if hasattr(attn,'_hf_hook'):attn._old_forward=types.MethodType(segmented_vision_forward,attn)
   else:attn.forward=types.MethodType(segmented_vision_forward,attn)
  for block in self.model.modules():
   if block.__class__.__name__=='PatchEmbed' and isinstance(getattr(block,'proj',None),torch.nn.Conv3d):
    def forward(self,x):
     w=self.proj.weight
     return torch.nn.functional.linear(x.reshape(-1,w[0].numel()).to(w.dtype),w.flatten(1),self.proj.bias)
    block.forward=types.MethodType(forward,block)
  self.processor=FlashVStreamQwen2VLProcessor.from_pretrained(p,local_files_only=True)
  self.processor.image_processor.max_pixels=PIXELS
  self.memory=dict(DEFAULT_FLASH_MEMORY_CONFIG)
  self.memory.update(self.model.config.vision_config.flash_memory_config)
 def answer(self,frames,meta,prompt,video):
  import torch
  p=self.processor
  text=p.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}],tokenize=False,add_generation_prompt=True)
  text+='Best option: ('  # Official inference_mcq_vqa.py assistant prefix, no option restriction.
  inputs=p(text=[text],videos=[frames],return_tensors='pt',padding=True,flash_memory_config=self.memory)
  info=record_inputs(inputs);device=self.model.get_input_embeddings().weight.device
  inputs={k:v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()}
  with torch.inference_mode(): out=self.model.generate(**inputs,max_new_tokens=128,do_sample=False)
  ids=out[0,inputs['input_ids'].shape[-1]:]
  return dict(response=p.batch_decode([ids],skip_special_tokens=True)[0].strip(),output_tokens=len(ids),flash_memory_config=self.memory,**info)
