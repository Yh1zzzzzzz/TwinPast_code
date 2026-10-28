# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
from common import *
class Adapter:
 def __init__(self):
  import sys,importlib.util,torch
  up=ADAPTERS / 'eval_dispider/upstream';sys.path.insert(0,str(up))
  spec=importlib.util.spec_from_file_location('dispider_official_inference',up/'inference.py')
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  self.native=module
  native_loader=module.load_pretrained_model
  from placement import load_distributed
  def load_on_two_gpus(*args,**kwargs):
   return load_distributed(native_loader,*args,**kwargs)
  module.load_pretrained_model=load_on_two_gpus
  self.runtime=module.VideoStream(str(MODEL_ROOT / 'Dispider'))
  self.model=self.runtime.model.eval()
 def answer(self,frames,meta,prompt,video):
  import torch,numpy as np
  from PIL import Image
  n=self.native;r=self.runtime
  assert len(frames)==64, 'Dispider requires complete 16-frame clips; do not silently pad'
  old=n.load_video
  def fixed(*args,**kwargs):
   # Keep the paper/native 16-frame clip grouping; substitute only uniform sampling.
   times=np.array([meta['timestamps_s'][i] for i in [0,16,32,48]]+
                  [meta['timestamps_s'][i]+1/meta['source_fps'] for i in [15,31,47,63]])
   native_frames=frames
   h,w=frames.shape[1:3]
   if h!=w:
    tensor=torch.from_numpy(frames).permute(0,3,1,2).float()
    tensor=torch.nn.functional.interpolate(tensor,size=(min(h,w),min(h,w)))
    native_frames=tensor.permute(0,2,3,1).to(torch.uint8).numpy()
   return [Image.fromarray(a) for a in native_frames],times,4
  n.load_video=fixed
  try:
   ids,images,large,seqs,mask,qs,qmask=n.process_data(video,[],prompt,self.model.config,r.tokenizer,r.image_processor,r.image_processor_large,r.time_tokenizer)
  finally: n.load_video=old
  inp={'input_ids':ids.unsqueeze(0).to(r.device),'images':images.to(r.device,dtype=torch.float16),
   'images_large':large.to(r.device,dtype=torch.float16),'seqs':seqs.to(r.device),'compress_mask':mask.to(r.device),
   'qs':qs.to(r.device),'qs_mask':qmask.to(r.device),'ans_token':r._ans_token.to(r.device),'todo_token':r._todo_token.to(r.device)}
  info=record_inputs(inp)
  info['text_input_tokens_before_expansion']=info['input_tokens']
  native_prepare=self.model.prepare_inference_inputs
  def trace_prepare(*args,**kwargs):
   prepared=native_prepare(*args,**kwargs)
   if prepared[-1] is not None:
    info['input_tokens']=int(prepared[-1].shape[1])
    info['multimodal_embedding_shape']=list(prepared[-1].shape)
   return prepared
  self.model.prepare_inference_inputs=trace_prepare
  try:
   with torch.inference_mode(): out=self.model.generate(**inp,q_id=None,insert_position=0,ans_position=[],do_sample=False,max_new_tokens=128,
        pad_token_id=r.tokenizer.eos_token_id,stopping_criteria=r.stopping_criteria,use_cache=True)
  finally: self.model.prepare_inference_inputs=native_prepare
  raw=r.tokenizer.batch_decode(out,skip_special_tokens=False)[0].strip()
  respond=n.DEFAULT_SILENT_TOKEN not in raw
  text=r.tokenizer.batch_decode(out,skip_special_tokens=True)[0].strip() if respond else ''
  return dict(response=text,raw_response=raw,
    should_respond=respond,output_tokens=out.shape[-1],weight_dtype='float16_native_dispider',native_preprocess='official square interpolation followed by CLIPImageProcessor',**info)
