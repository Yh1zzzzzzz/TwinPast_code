# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
from common import *
class Adapter:
 def __init__(self):
  from transformers import AutoModelForCausalLM,AutoProcessor
  p=str(MODEL_ROOT / 'MOSS-VL-Realtime')
  self.processor=AutoProcessor.from_pretrained(p,trust_remote_code=True,local_files_only=True)
  self.model=AutoModelForCausalLM.from_pretrained(p,**load_kwargs()).eval()
  fast_patch(self.model)
 def answer(self,frames,meta,prompt,video):
  import torch,types
  p=self.processor; vp=p.video_processor; old=vp.fetch_videos
  def fixed(self, videos,**kwargs):
   return [torch.from_numpy(frames).permute(0,3,1,2)],[metadata(meta)]
  vp.fetch_videos=types.MethodType(fixed,vp)
  query={'prompt':prompt,'images':[],'videos':[video],
   'media_kwargs':{'video_max_pixels':64*PIXELS,'min_frames':len(frames),'max_frames':len(frames)},
   'generate_kwargs':{'max_new_tokens':128,'do_sample':False},'reset_session':True,'persist_session':False}
  info={}; original=self.model.generate
  def traced(*a,**kw):
   info.update(record_inputs(kw)); out=original(*a,**kw)
   info['output_tokens']=int(out.shape[-1]-kw['input_ids'].shape[-1]); return out
  self.model.generate=traced
  try:
   with torch.inference_mode(): res=self.model.offline_batch_generate(p,[query],session_states=[[]],vision_chunked_length=64)
  finally: vp.fetch_videos=old; self.model.generate=original
  return dict(response=res['results'][0]['text'].strip(),**info)
