# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Official JoyAI chunk/memory controller with local HF model transports."""
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
import asyncio,base64,io,os,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
from common import ROOT,load_kwargs,fast_patch,record_inputs
from native import CausalFrames


class LocalModel:
    def __init__(self,model,processor,skip_special):
        self.model=model;self.processor=processor;self.skip_special=skip_special
        self.lock=threading.Lock();self.records=[]

    def create(self,**kwargs):
        import torch
        from PIL import Image
        messages=[];images=[]
        for message in kwargs['messages']:
            content=message.get('content','')
            if isinstance(content,list):
                parts=[]
                for item in content:
                    kind=item.get('type')
                    if kind=='image_url':
                        url=item['image_url'];url=url['url'] if isinstance(url,dict) else url
                        if not url.startswith('data:image/'):
                            raise ValueError('Local evaluation accepts only controller-encoded local images')
                        im=Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))).convert('RGB')
                        parts.append(dict(type='image',image=im));images.append(im)
                    elif kind=='text':parts.append(dict(type='text',text=item['text']))
                    else:raise ValueError('Unsupported native JoyAI message content: '+str(kind))
                content=parts
            messages.append(dict(role=message['role'],content=content))
        with self.lock,torch.inference_mode():
            p=self.processor
            text=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=p(text=[text],images=images or None,return_tensors='pt',padding=True)
            device=self.model.get_input_embeddings().weight.device
            inputs={k:v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()}
            start=time.monotonic();out=self.model.generate(**inputs,max_new_tokens=kwargs['max_tokens'],do_sample=False)
            ids=out[0,inputs['input_ids'].shape[1]:]
            raw=p.tokenizer.decode(ids,skip_special_tokens=self.skip_special).strip()
            self.records.append(dict(**record_inputs(inputs),output_tokens=len(ids),wall_s=time.monotonic()-start,
                                     response=raw,model_class=type(self.model).__name__))
        usage=dict(prompt_tokens=inputs['input_ids'].shape[1],completion_tokens=len(ids),total_tokens=out.shape[1])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))],
                               usage=SimpleNamespace(model_dump=lambda:usage))


class Adapter:
    def __init__(self):
        import torch
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        compact=os.environ.get('JOYAI_PLACEMENT')=='two_gpu'
        if torch.cuda.device_count()<(2 if compact else 3):raise RuntimeError('Not enough reserved GPUs for the selected JoyAI placement')
        up=ADAPTERS / 'eval_joyai_vl/upstream/services/webinfer';sys.path.insert(0,str(up))
        import live_adapter
        self.native=live_adapter
        main=str(MODEL_ROOT / 'JoyAI-VL-Interaction');summary=str(MODEL_ROOT / 'Qwen3-VL-4B')
        kw=load_kwargs()
        if compact:
            # Keep main sampling on its input device0 and summary sampling on
            # device1, preserving independent per-role CUDA RNG streams.
            # Move the last 8 main layers to device1; leave room there for 4B
            # summary weights, and keep the remaining main/vision weights on0.
            kw.pop('max_memory',None)
            # Use a complete, non-overlapping map. A root entry plus child
            # overrides can give Accelerate conflicting execution-device hooks.
            kw['device_map']={'model.visual':0,'model.language_model.embed_tokens':0,
                             **{f'model.language_model.layers.{i}':0 if i<28 else 1 for i in range(36)},
                             'model.language_model.norm':1,'model.language_model.rotary_emb':0,'lm_head':1}
        else:kw['max_memory']={0:'20GiB',1:'20GiB'}
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(main,**kw).eval();fast_patch(self.model)
        self.processor=AutoProcessor.from_pretrained(main,local_files_only=True)
        summary_model=Qwen3VLForConditionalGeneration.from_pretrained(summary,local_files_only=True,
                      torch_dtype=torch.bfloat16,device_map={'':1 if compact else 2},attn_implementation='sdpa').eval();fast_patch(summary_model)
        summary_processor=AutoProcessor.from_pretrained(summary,local_files_only=True)
        for processor in [self.processor,summary_processor]:
            processor.image_processor.size={'shortest_edge':4096,'longest_edge':262144}
        self.main=LocalModel(self.model,self.processor,False);self.summary=LocalModel(summary_model,summary_processor,True)
        self.root=RUN_ROOT / 'joyai_frames'
        # Only localhost client objects are constructed by the upstream controller;
        # every completion call is replaced with an in-process transport below.
        cfg=live_adapter.AdapterConfig(chunk=100,main_model=main,summarizer_model=summary,longterm_model=summary,
              main_temperature=0,main_max_tokens=128,mid_term_temperature=0,long_term_temperature=0,
              frame_save_dir=str(self.root),allowed_local_image_roots=(str(self.root),),
              save_model_inputs=False,save_debug_inputs=False,async_summary_lead_frames=10)
        self.controller=live_adapter.StreamingInferAdapter(cfg)
        async def main_create(**kwargs):return self.main.create(**kwargs)
        self.controller.main_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=main_create)))
        summarizer=self.controller.summarizer
        def summary_chat(messages,max_tokens,**kwargs):
            return self.summary.create(messages=messages,max_tokens=max_tokens).choices[0].message.content
        summarizer._chat=summary_chat;summarizer._tokenizer=summary_processor.tokenizer

    def answer_stream(self,job,prompt,emit):
        return asyncio.run(self._answer_stream(job,prompt,emit))

    async def _answer_stream(self,job,prompt,emit):
        import torch
        controller=self.controller;state=controller.get_session(job['job_id'])
        frames=CausalFrames(job,fps=1.0);folder=self.root/job['job_id'];folder.mkdir(parents=True,exist_ok=True)
        main_start=len(self.main.records);summary_start=len(self.summary.records);start=time.monotonic();count=0;last_path=None
        request=SimpleNamespace(headers={})
        try:
            for image,timestamp in frames:
                count+=1;last_path=folder/f'{round(timestamp*frames.source_fps):08d}.png'
                if not last_path.exists():image.save(last_path)
                result=await controller._handle_chat_payload(state,dict(messages=[dict(role='user',content='')],
                       local_image_path=str(last_path),frame_time_range=f'{timestamp:.1f} seconds'),request)
                emit(dict(type='frame',timestamp_s=timestamp,frames_received=count,chunk_index=state.chunk_index,
                          retained_current_frames=state.current_chunk['frame_count'],mid_term_summaries=len(state.mid_term_summaries),
                          response=result['choices'][0]['message']['content']))
            # Complete all required memory summaries before asking. No question
            # is passed to the summarizer during historical observation.
            await controller._commit_required_async_summaries(state,state.turn_count,non_blocking=False)
            ingest_s=time.monotonic()-start;query_start=time.monotonic()
            emit(dict(type='query',timestamp_s=job['t_query'],frames_received=count,chunk_index=state.chunk_index,
                      mid_term_memory=state.mid_term_summaries,long_term_memory=state.memory_state['long_term_memory']))
            # The released controller handles user queries with a contemporaneous
            # frame; reuse the last observed image, never fetch a future frame.
            result=await controller._handle_chat_payload(state,dict(messages=[dict(role='user',content=prompt)],
                   local_image_path=str(last_path),frame_time_range=f"{job['t_query']:.3f} seconds"),request)
            for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
            raw=result['choices'][0]['message']['content'];response=self.native.extract_response_payload(raw) or ''
            calls=self.main.records[main_start:];summaries=self.summary.records[summary_start:]
            for call in calls:emit(dict(type='main_inference',**call))
            for call in summaries:emit(dict(type='summary_inference',**call))
            return dict(response=response,raw_response=raw,output_tokens=calls[-1]['output_tokens'] if calls else 0,
                    input_tokens=calls[-1]['input_tokens'] if calls else None,sampling=frames.meta,
                    frames_received=count,frames_processed=state.current_chunk['frame_count'],ingestion_wall_s=ingest_s,
                    query_wall_s=time.monotonic()-query_start,native_api='official StreamingInferAdapter chunk + memory controller',
                    chunk_frames=100,summary_model='Qwen3-VL-4B-Instruct',summary_calls=len(summaries),
                    mid_term_summaries=state.mid_term_summaries,long_term_memory=state.memory_state['long_term_memory'],
                    gate='official force_silence_before_query default',transport='local HF replacement for vLLM; no external delegate invoked')
        finally:
            for pending in state.async_pending_summary_jobs:pending['task'].cancel()
            controller.sessions.pop(job['job_id'],None)
