# SPDX-License-Identifier: Apache-2.0
import copy, time
from .config import MODEL_ROOT, CONDITIONS, stat
from .base import TransformersBackend
from .media import portable_messages

class Backend(TransformersBackend):
    def __init__(self):
        super().__init__(MODEL_ROOT/'AURA-8B',128,20)
        self.metadata.update(fps_by_condition={'tail30':1.,'streaming':2.},fps='condition-specific')
        self.cache={};self.fetch=self.processor.video_processor.fetch_videos
        self.processor.video_processor.fetch_videos=self.fetch_cached
        self.sampled=[];self.fps=1.;self.active=set()

    def reset_session(self,condition):
        self.cache.clear();self.fps=1. if condition==CONDITIONS[0] else 2.

    def fetch_cached(self,video,sample_indices_fn=None):
        if not isinstance(video,str):return self.fetch(video,sample_indices_fn=sample_indices_fn)
        identity=stat(video)
        if video in self.cache and self.cache[video][0]==identity:
            _,frames,meta=self.cache[video];meta=copy.deepcopy(meta)
        else:
            frames,meta=self.fetch(video,sample_indices_fn=sample_indices_fn)
            self.cache[video]=(identity,frames,copy.deepcopy(meta))
        self.sampled.append(dict(path=video,frames_indices=list(map(int,meta.frames_indices)),fps=float(meta.fps),shape=list(frames.shape)))
        return frames,meta

    def generate(self,messages):
        torch=self.torch;start=time.monotonic();messages=portable_messages(messages)
        videos=[i['video'] for m in messages if isinstance(m['content'],list) for i in m['content'] if i.get('type')=='video']
        self.active=set(videos);self.cache={p:v for p,v in self.cache.items() if p in self.active};self.sampled=[]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch=self.processor(text=[text],videos=videos,fps=self.fps,size={'shortest_edge':4096,'longest_edge':602112},return_tensors='pt')
        shapes={k:list(v.shape) for k,v in batch.items()};n=int(batch['input_ids'].shape[1])
        batch=batch.to(self.model.get_input_embeddings().weight.device)
        for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i);torch.cuda.reset_peak_memory_stats(i)
        prepared=time.monotonic()-start;inference=time.monotonic()
        kwargs={}
        if self.fps==2.:
            from transformers import StoppingCriteria,StoppingCriteriaList
            class Silence(StoppingCriteria):
                def __call__(self,input_ids,scores,**kw):
                    return (input_ids[:,n]==151669)|(input_ids[:,n]==151645)
            kwargs['stopping_criteria']=StoppingCriteriaList([Silence()])
        with torch.inference_mode():
            result=self.model.generate(**batch,do_sample=False,max_new_tokens=128,use_cache=True,**kwargs)
        for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
        ids=result[0,n:].detach().cpu().tolist()
        raw=self.processor.tokenizer.decode(ids,skip_special_tokens=False)
        response=raw.removesuffix('<|im_end|>').removesuffix('<|endoftext|>').strip()
        if self.fps==2. and ids and ids[0] in (151669,151645):response='<|silent|>'
        return dict(response=response,raw_decoded_response=raw,generated_ids=ids,input_tokens=n,output_tokens=len(ids),
            prepare_seconds=prepared,inference_seconds=time.monotonic()-inference,wall_seconds=time.monotonic()-start,
            input_shapes=shapes,sampled_videos=self.sampled,
            peak_allocated_mib=[torch.cuda.max_memory_allocated(i)/1048576 for i in range(torch.cuda.device_count())],
            peak_reserved_mib=[torch.cuda.max_memory_reserved(i)/1048576 for i in range(torch.cuda.device_count())],
            execution='HF full active-context recomputation, no cross-step KV; active media decode cache only')
