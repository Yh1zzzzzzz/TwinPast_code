# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Released realtime CSM/DAM memory API, one causal frame per update."""
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
import sys,time,types
from common import ROOT,load_kwargs,record_inputs
from native import CausalFrames
from adapter import segmented_vision_forward


class Adapter:
    def __init__(self):
        import torch
        up=ADAPTERS / 'eval_flash_vstream/upstream/Flash-VStream-Qwen'
        sys.path.insert(0,str(up))
        from models.vstream_qwen2vl_realtime import FlashVStreamQwen2VLModel
        from models import FlashVStreamQwen2VLProcessor,DEFAULT_FLASH_MEMORY_CONFIG
        path=str(MODEL_ROOT / 'Flash-VStream-Qwen-7b')
        self.model=FlashVStreamQwen2VLModel.from_pretrained(path,**load_kwargs()).eval()
        for block in self.model.visual.blocks:
            attn=block.attn
            if hasattr(attn,'_hf_hook'):attn._old_forward=types.MethodType(segmented_vision_forward,attn)
            else:attn.forward=types.MethodType(segmented_vision_forward,attn)
        for block in self.model.modules():
            if block.__class__.__name__=='PatchEmbed' and isinstance(getattr(block,'proj',None),torch.nn.Conv3d):
                def forward(self,x):
                    w=self.proj.weight
                    return torch.nn.functional.linear(x.reshape(-1,w[0].numel()).to(w.dtype),w.flatten(1),self.proj.bias)
                block.forward=types.MethodType(forward,block)
        self.processor=FlashVStreamQwen2VLProcessor.from_pretrained(path,local_files_only=True)
        # The realtime CLI's source loader caps frames at 4*224*224 pixels.
        self.processor.image_processor.max_pixels=4*224*224
        self.memory=dict(DEFAULT_FLASH_MEMORY_CONFIG)
        self.memory.update(self.model.config.vision_config.flash_memory_config)
        self.model.use_video_streaming_mode=True
        self.model.video_embedding_memory=[]

    def answer_stream(self,job,prompt,emit):
        import torch,numpy as np
        model=self.model;processor=self.processor;model.video_embedding_memory=[]
        frames=CausalFrames(job,fps=1.0);start=time.monotonic();count=0
        try:
            with torch.inference_mode():
                for image,timestamp in frames:
                    values=processor.image_processor(images=None,videos=[[np.asarray(image)]],return_tensors='pt',
                        additional_pool_size=self.memory['flash_memory_temporal_poolsize'])
                    tick=time.monotonic()
                    model.embed_new_video_clip(**values,start_idx=count)
                    for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
                    count+=1;m=model.video_embedding_memory
                    emit(dict(type='frame',timestamp_s=timestamp,frames_processed=count,update_s=time.monotonic()-tick,
                              temporal_thw=m[1].tolist(),spatial_thw=m[5].tolist(),visual_tokens=int(m[-1][0]),
                              input_shapes={k:list(v.shape) for k,v in values.items()}))
                ingest_s=time.monotonic()-start;query_start=time.monotonic()
                visual_tokens=int(model.video_embedding_memory[-1][0])
                emit(dict(type='query',timestamp_s=job['t_query'],frames_processed=count,visual_tokens=visual_tokens))
                text=processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':
                      '<|vision_start|><|video_pad|><|vision_end|>'+prompt}]}],tokenize=False,add_generation_prompt=True)
                text+='Best Option: ('
                inputs=processor(text=[text],videos=None,images=None,padding=True,return_tensors='pt',
                                 flash_memory_config=self.memory,dummy_video_tokens=visual_tokens*4)
                info=record_inputs(inputs);device=model.get_input_embeddings().weight.device
                inputs={k:v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()}
                # Ingestion leaves large inactive allocator blocks. Release only
                # unused blocks before HF clones the full prefill logits; the
                # tensors, memory policy, precision and generation are unchanged.
                allocator=[]
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        before=torch.cuda.memory_reserved(i)
                        torch.cuda.empty_cache()
                        allocator.append(dict(device=i,reserved_before=before,
                            reserved_after=torch.cuda.memory_reserved(i),allocated=torch.cuda.memory_allocated(i)))
                emit(dict(type='allocator_trim_before_query',devices=allocator))
                output=model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True)
                for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
                ids=output[0,inputs['input_ids'].shape[1]:]
                return dict(response=processor.batch_decode([ids],skip_special_tokens=True)[0].strip(),output_tokens=len(ids),
                            sampling=frames.meta,frames_processed=count,ingestion_wall_s=ingest_s,
                            query_wall_s=time.monotonic()-query_start,native_api='embed_new_video_clip + prepare_realtime_inference',
                            flash_memory_config=self.memory,visual_memory_tokens=visual_tokens,**info)
        finally:model.video_embedding_memory=[]
