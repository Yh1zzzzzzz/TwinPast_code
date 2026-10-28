# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
from common import *
class Adapter:
 def __init__(self):
  import sys,torch
  from functools import partial
  from transformers import AutoModel
  sys.path.insert(0,str(ADAPTERS / 'eval_videollm_online/upstream'))
  from models import build_model_and_tokenizer
  from models.vision_live import _siglip_vision_encode
  self.model,self.processor=build_model_and_tokenizer(is_training=False,set_vision_inside=False,
    llm_pretrained=str(MODEL_ROOT / 'Meta-Llama-3-8B-Instruct'),resume_from_checkpoint=str(MODEL_ROOT / 'videollm-online-8b-v1plus'),
    vision_pretrained='google/siglip-large-patch16-384',frame_resolution=384,frame_token_cls=True,
    frame_token_pooled=[3,3],frame_num_tokens=10,frame_token_interval=',',torch_dtype=torch.bfloat16,attn_implementation='sdpa')
  self.model=self.model.to('cuda:0').eval()
  vd='cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
  vision=AutoModel.from_pretrained(str(MODEL_ROOT / 'siglip-large-patch16-384'),local_files_only=True,torch_dtype=torch.float32).vision_model.to(vd).eval()
  backbone=self.model.get_base_model()
  backbone.vision_encoder=vision
  def encode(encoder,frames):
   return torch.cat([_siglip_vision_encode(encoder,x.to(vd),frame_token_cls=True,frame_token_pooled=[3,3])
                     for x in frames.split(4)]).to('cuda:0')
  backbone.vision_encode=encode
 def answer(self,frames,meta,prompt,video):
  import torch
  tok=self.processor
  messages=[{'role':'system','content':'A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant.'},
    {'role':'stream','num_frames':len(frames)},{'role':'user','content':prompt}]
  ids=tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors='pt').to('cuda:0')
  from preprocessing import resize_and_pad
  rgb=resize_and_pad(frames,resolution=384).to('cuda:0')
  with torch.inference_mode(): out=self.model.generate_after_embed(ids,rgb,attention_mask=torch.ones_like(ids),max_new_tokens=128,do_sample=False,use_cache=True,
    eos_token_id=self.model.config.eos_token_id,pad_token_id=tok.eos_token_id)
  return dict(response=tok.decode(out[0],skip_special_tokens=True).strip(),output_tokens=out.shape[-1],input_tokens=ids.shape[-1],
    input_shapes={'frames':list(rgb.shape)},frame_visual_tokens=10,input_mode='offline_fullprefix_stream_template')
