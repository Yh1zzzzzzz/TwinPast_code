# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Official ReKV LLaVA-OneVision 7B; query-independent encoding, internal retrieval."""
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
import torch
import numpy as np
import os
from native import CausalFrames

class Adapter:
    def __init__(self):
        from model.llava_onevision_rekv import load_model
        self.model,self.processor=load_model(str(MODEL_ROOT / 'llava-onevision-qwen2-7b-ov-hf'),n_local=15000,topk=64,chunk_size=1)
        self.model.language_model.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
        self.selected_vision_state_only=os.environ.get('REKV_SELECTED_VISION_STATE_ONLY')=='1'
        if self.selected_vision_state_only:
            tower=self.model.vision_tower;layers=tower.vision_model.encoder.layers
            index=self.model.config.vision_feature_layer%(len(layers)+1)
            original=tower.forward
            def vision_forward(*args,**kwargs):
                captured=[]
                def capture(module,inputs,output):captured.append(output[0] if isinstance(output,tuple) else output)
                def capture_input(module,inputs):captured.append(inputs[0])
                hook=layers[index-1].register_forward_hook(capture) if index else layers[0].register_forward_pre_hook(capture_input)
                try:
                    # Same64-frame operations and FP16 kernels. Release unused
                    # encoder outputs instead of accumulating every layer.
                    kwargs['output_hidden_states']=False
                    result=original(*args,**kwargs)
                finally:hook.remove()
                states=[None]*(len(layers)+1);states[index]=captured[0]
                result.hidden_states=tuple(states)
                return result
            tower.forward=vision_forward

    @torch.inference_mode()
    def answer_stream(self,job,question,emit):
        stream=CausalFrames(job,fps=0.5);emit(dict(type='sampling',**stream.meta))
        self.model.clear_cache();self.model.encode_init_prompt()
        pending=[];count=0
        def consume():
            nonlocal count
            video=torch.from_numpy(np.stack(pending));self.model.encode_video(video,encode_chunk_size=64)
            count+=len(pending);pending.clear()
        for image,t in stream:
            pending.append(np.asarray(image))
            if len(pending)==64:consume();emit(dict(type='observe',timestamp_s=t,frames=count))
        if pending:consume()
        cpu_bytes=self.model.calc_memory_usage()
        emit(dict(type='memory_ready',frames=count,query_seen=False,cpu_kv_bytes=cpu_bytes))
        # The runner sends only the raw question to internal retrieval, as in BaseVQA.
        raw_question=job['question']
        response=self.model.question_answering(dict(question=raw_question,prompt=self.model.get_prompt(question,mc=True)),max_new_tokens=256)
        emit(dict(type='answer',response=response))
        self.model.clear_cache()
        return dict(response=response,frames=count,cpu_kv_bytes=cpu_bytes,n_local=15000,retrieve_frames=64,retrieve_chunk_size=1,readout='free_generation',generation_limit=256,selected_vision_state_only=self.selected_vision_state_only)
