# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Released proactive controller with local HF transport and launch-script settings.

Presence penalty counts generated tokens only, matching OpenAI/vLLM semantics.
The controller, prompts, response parsing, chunking and summaries stay upstream.
"""
import base64,io,threading,time,os
from types import SimpleNamespace
from common import record_inputs
from streaming_adapter import Adapter as ControlAdapter


def generation_kwargs(kwargs,prompt_length):
    from transformers import LogitsProcessor,LogitsProcessorList
    extra=kwargs.get('extra_body',{})
    temperature=float(kwargs.get('temperature',.8))
    penalty=float(kwargs.get('presence_penalty',0.0))
    class GeneratedPresencePenalty(LogitsProcessor):
        def __call__(self,input_ids,scores):
            for row in range(input_ids.shape[0]):
                seen=input_ids[row,prompt_length:].unique()
                scores[row,seen]-=penalty
            return scores
    result=dict(max_new_tokens=kwargs['max_tokens'],do_sample=temperature>0,
                repetition_penalty=extra.get('repetition_penalty',kwargs.get('repetition_penalty',1.0)))
    if temperature>0:result.update(temperature=temperature,top_p=kwargs.get('top_p',.9),top_k=max(0,extra.get('top_k',kwargs.get('top_k',0))))
    if penalty:result['logits_processor']=LogitsProcessorList([GeneratedPresencePenalty()])
    return result

class NativeLocalModel:
    def __init__(self,model,processor,skip_special,use_cache=None):
        self.model=model;self.processor=processor;self.skip_special=skip_special
        self.lock=threading.Lock();self.records=[];self.use_cache=use_cache

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
            generation=generation_kwargs(kwargs,inputs['input_ids'].shape[1])
            if self.use_cache is not None:generation['use_cache']=self.use_cache
            start=time.monotonic();out=self.model.generate(**inputs,**generation)
            ids=out[0,inputs['input_ids'].shape[1]:]
            raw=p.tokenizer.decode(ids,skip_special_tokens=self.skip_special).strip()
            self.records.append(dict(**record_inputs(inputs),output_tokens=len(ids),output_token_ids=ids.tolist(),wall_s=time.monotonic()-start,
                                     response=raw,model_class=type(self.model).__name__,generation_parameters={**{k:v for k,v in generation.items() if k!='logits_processor'},'presence_penalty':kwargs.get('presence_penalty',0.0)}))
        usage=dict(prompt_tokens=inputs['input_ids'].shape[1],completion_tokens=len(ids),total_tokens=out.shape[1])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))],
                               usage=SimpleNamespace(model_dump=lambda:usage))


class Adapter(ControlAdapter):
    def __init__(self):
        super().__init__()
        self.main=NativeLocalModel(self.main.model,self.main.processor,False)
        self.summary=NativeLocalModel(self.summary.model,self.summary.processor,True)
        # Cache-only continuation keeps existing scored rows; each new runtime
        # and answer explicitly records this optimization/version boundary.
        if os.environ.get('JOYAI_GENERATION_USE_CACHE')=='1':
            self.main.use_cache=True;self.summary.use_cache=True
        # Official scripts/start_adapter.sh settings, except the declared common
        # 128-token final Query budget. Background main calls retain 256 tokens.
        settings=dict(force_silence_before_query=False,chunk=100,
            main_max_tokens=256,main_temperature=.8,main_top_p=.9,main_top_k=40,
            main_repetition_penalty=1.05,main_presence_penalty=.4,
            mid_term_temperature=.8,mid_term_top_p=.9,mid_term_top_k=40,
            mid_term_repetition_penalty=1.1,mid_term_presence_penalty=1.0,
            long_term_temperature=.3,long_term_top_p=.7,long_term_top_k=30,
            long_term_repetition_penalty=1.1,long_term_presence_penalty=.5,
            long_term_max_tokens=4000,long_term_target_tokens=2000,long_term_memory_window=15)
        for key,value in settings.items():
            setattr(self.controller.config,key,value)
            if hasattr(self.controller.summarizer,key):setattr(self.controller.summarizer,key,value)
        original_generation=self.controller._main_generation_kwargs
        def generation_for_payload(payload):
            result=original_generation(payload)
            is_query=any(m.get('role')=='user' and bool(m.get('content')) for m in payload.get('messages',[]))
            result['max_tokens']=128 if is_query else 256
            return result
        self.controller._main_generation_kwargs=generation_for_payload
        self.root=self.root.parent/'joyai_proactive'
        self.controller.config.frame_save_dir=str(self.root)
        self.controller.config.allowed_local_image_roots=(str(self.root),)
        def summary_chat(messages,max_tokens,**kwargs):
            kwargs.pop('client',None);kwargs.pop('model_name',None)
            return self.summary.create(messages=messages,max_tokens=max_tokens,**kwargs).choices[0].message.content
        self.controller.summarizer._chat=summary_chat

    async def _answer_stream(self,job,prompt,emit):
        begin=len(self.main.records)
        try:
            result=await super()._answer_stream(job,prompt,emit)
            result.update(main_use_cache=self.main.use_cache,summary_use_cache=self.summary.use_cache,
                cache_scope='per-generation only; no cross-request reuse',
                native_api='official proactive StreamingInferAdapter + chunk memory summaries',
                force_silence_before_query=False,history_main_calls=len(self.main.records)-begin-1,
                gate='released proactive response/silence/delegate controller',
                sampling_protocol='official start_adapter.sh sampling and penalties; local HF, not numerical vLLM equivalence',
                transport='local HF; no external delegate agent; background main cap256/final Query cap128, summaries launch-script budgets')
            return result
        finally:
            self.main.records.clear();self.summary.records.clear()
