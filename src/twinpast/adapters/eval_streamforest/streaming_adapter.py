# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Official StreamForest PEMF/FSTW, with causal incremental feature delivery.
The learned modules and MemoryManager are unmodified; upstream batch_size=128
vision forwarding is reduced to small independent frame batches on 24-GiB GPUs.
"""
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
import math
import torch
from native import CausalFrames

def update_memory(projector,manager,features,offset):
    # Same positional additions/order and 512-frame periodicity as official forward.
    spatial=projector.get_spatial_pos_embed(projector.hw,features.device).to(features.dtype)
    positions=(torch.arange(offset,offset+len(features),device=features.device)%projector.time_pos_embedding_window)
    temporal=projector.temporal_pos_embed[positions].reshape(len(features),1,-1).to(features.dtype)
    features=features+spatial+temporal
    for feature in features:manager.update(feature)

class Adapter:
    def __init__(self):
        from pathlib import Path
        from llava.model.builder import load_pretrained_model
        from llava.model.multimodal_projector.memory_manager import MemoryManager
        root=RELEASE_ROOT
        self.tokenizer,self.model,self.processor,_=load_pretrained_model(
            str(MODEL_ROOT/'StreamForest-Qwen2-7B'),None,'llava_qwen',multimodal=True,
            overwrite_config={'mm_vision_tower':str(MODEL_ROOT/'siglip-so400m-patch14-384')},
            device_map='cuda:0',attn_implementation='sdpa')
        self.model.eval();self.MemoryManager=MemoryManager
        # Only final-position logits are read during greedy inference.
        self.model.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))

    @torch.inference_mode()
    def answer_stream(self,job,question,emit):
        from llava.conversation import conv_templates,SeparatorStyle
        from llava.mm_utils import tokenizer_image_token,KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX
        from transformers import Qwen2ForCausalLM
        frames=CausalFrames(job,fps=1.0)
        emit(dict(type='sampling',**frames.meta))
        tower=self.model.get_vision_tower();projector=self.model.get_model().mm_projector
        manager=self.MemoryManager(projector.mm_hidden_size,projector.num_attention_heads).to(tower.device)
        pending=[];offset=0
        def consume():
            nonlocal offset
            pixel=self.processor.preprocess(pending,return_tensors='pt')['pixel_values'].to(tower.device,torch.float16)
            feats=tower(pixel)
            update_memory(projector,manager,feats,offset)
            offset+=len(pending);pending.clear()
        times=[]
        for image,t in frames:
            pending.append(image);times.append(t)
            if len(pending)==4:consume();emit(dict(type='observe',timestamp_s=t,frames=offset))
        if pending:consume()
        memory=projector.mlp(manager.get_memory_tokens())
        emit(dict(type='memory_ready',timestamp_s=times[-1],frames=offset,memory_tokens=memory.shape[-2],query_seen=False))
        msg=f'\nThe video contains {len(times)} frames sampled from the past {times[-1]-times[0]:.1f} seconds ago ({times[0]:.1f}s of the entire video) up to the present moment ({times[-1]:.1f}s of the entire video). '
        conv=conv_templates['qwen_2'].copy();conv.append_message(conv.roles[0],'<image>\n'+msg.rstrip()+' '+question);conv.append_message(conv.roles[1],None)
        prompt=conv.get_prompt();ids=tokenizer_image_token(prompt,self.tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop=conv.sep if conv.sep_style!=SeparatorStyle.TWO else conv.sep2
        dummy=torch.empty((1,3,384,384),device=self.model.device,dtype=torch.float16)
        _,positions,mask,_,embeds,_=self.model.prepare_inputs_labels_for_LLM(ids,None,None,None,None,[dummy],[memory],modalities=['video'])
        out=Qwen2ForCausalLM.generate(self.model,inputs_embeds=embeds,position_ids=positions,attention_mask=mask,
            do_sample=False,max_new_tokens=1024,num_beams=1,use_cache=True,pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stopping_criteria=[KeywordsStoppingCriteria([stop],self.tokenizer,ids)])
        response=self.tokenizer.batch_decode(out,skip_special_tokens=True)[0].strip()
        emit(dict(type='answer',response=response))
        return dict(response=response,frames=offset,memory_tokens=memory.shape[-2],input_tokens=embeds.shape[1],readout='free_generation',generation_limit=1024)
