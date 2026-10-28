# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
from common import *
class Adapter:
 def __init__(self):
  # This vision-only run never invokes audio feature extraction.
  import transformers.utils.import_utils as iu
  iu._librosa_available=False
  from transformers import AutoModel
  from transformers.cache_utils import Cache
  if not hasattr(Cache,'seen_tokens'): Cache.seen_tokens=property(lambda self:self.get_seq_length())
  p=str(MODEL_ROOT / 'MiniCPM-o-4_5')
  kw=load_kwargs()
  # Keep visual/resampled embeddings with the token embeddings for native scatter_.
  kw['device_map']={'llm.model.embed_tokens':0,'vpm':0,'resampler':0,
      'llm.model.norm':1,'llm.lm_head':1,**{f'llm.model.layers.{i}':0 if i<14 else 1 for i in range(json.loads((MODEL_ROOT / 'MiniCPM-o-4_5/config.json').read_text())['num_hidden_layers'])}}
  self.model=AutoModel.from_pretrained(p,init_vision=True,init_audio=False,init_tts=False,**kw).eval()
 def answer(self,frames,meta,prompt,video):
  import torch
  from PIL import Image
  self.model.reset_session()
  info={}; original=self.model.generate
  def traced(*a,**kw):
   info.update(record_inputs(kw)); out=original(*a,**kw)
   info['output_tokens']=int(out[1].sequences.shape[-1]); return out
  self.model.generate=traced
  try:
   with torch.inference_mode():
    text=self.model.chat(msgs=[{'role':'user','content':[Image.fromarray(a) for a in frames]+[prompt]}],
       do_sample=False,num_beams=1,max_new_tokens=128,max_inp_length=32768,max_slice_nums=1,use_image_id=False,
       enable_thinking=False,generate_audio=False,use_tts_template=False)
  finally: self.model.generate=original
  return dict(response=text,**info)
