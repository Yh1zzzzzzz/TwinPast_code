# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
from common import *
class Adapter:
 def __init__(self):
  import torch
  from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
  p=str(MODEL_ROOT / 'JoyAI-VL-Interaction')
  self.processor=AutoProcessor.from_pretrained(p,local_files_only=True)
  self.model=Qwen3VLForConditionalGeneration.from_pretrained(p,**load_kwargs()).eval()
  fast_patch(self.model)
  self.processor.video_processor.max_video_tokens=round(PIXELS/1024)
  self.processor.video_processor.size={'shortest_edge':4096,'longest_edge':25165824}
 def answer(self,frames,meta,prompt,video):
  import torch
  p=self.processor
  msg=[{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}]
  text=p.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
  inputs=p(text=[text],videos=[frames],video_metadata=[metadata(meta)],do_sample_frames=False,
           cap_pixels_per_frame=True,return_tensors='pt')
  info=record_inputs(inputs); device=self.model.get_input_embeddings().weight.device
  inputs={k:v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()}
  with torch.inference_mode(): out=self.model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True)
  ids=out[0,inputs['input_ids'].shape[1]:]
  return dict(response=p.tokenizer.decode(ids,skip_special_tokens=True).strip(),output_tokens=len(ids),**info)
