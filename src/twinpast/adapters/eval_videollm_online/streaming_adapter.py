# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Official LiveInfer streaming KV loop with an end-of-stream query flush."""
import time
from adapter import Adapter as OfflineAdapter
from native import CausalFrames


class Adapter(OfflineAdapter):
    def __init__(self):
        super().__init__()
        import torch
        from demo.inference import LiveInfer
        live=LiveInfer.__new__(LiveInfer);live.model=self.model;live.tokenizer=self.processor
        cfg=self.model.config
        live.hidden_size=cfg.hidden_size;live.frame_fps=2.;live.frame_interval=.5
        live.frame_resolution=cfg.frame_resolution;live.frame_num_tokens=cfg.frame_num_tokens
        live.frame_token_interval_id=cfg.frame_token_interval_id
        live.system_prompt='A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant.'
        live.inplace_output_ids=torch.zeros(1,100,device='cuda:0',dtype=torch.long)
        live.frame_token_interval_threshold=.725;live.eos_token_id=cfg.eos_token_id
        live._start_ids=self.processor.apply_chat_template([dict(role='system',content=live.system_prompt)],add_stream_prompt=True,return_tensors='pt').to('cuda:0')
        live._added_stream_prompt_ids=self.processor.apply_chat_template([{}],add_stream_prompt=True,return_tensors='pt').to('cuda:0')
        live._added_stream_generation_ids=self.processor.apply_chat_template([{}],add_stream_generation_prompt=True,return_tensors='pt').to('cuda:0')
        self.live=live

    def answer_stream(self,job,prompt,emit):
        import torch,numpy as np
        live=self.live;live.reset();live.inplace_output_ids=torch.zeros(1,100,device='cuda:0',dtype=torch.long)
        frames=CausalFrames(job,fps=2.);start=time.monotonic();count=0
        with torch.inference_mode():
            for image,timestamp in frames:
                tick=time.monotonic()
                from preprocessing import resize_and_pad
                rgb=resize_and_pad(np.asarray(image),resolution=384).to('cuda:0')
                embeds=self.model.visual_embed(rgb)
                live.frame_embeds_queue.append((timestamp,embeds));count+=1
                result_time,query=live._call_for_streaming()
                response=None
                if result_time is not None:
                    _,response=live._call_for_response(result_time,query)
                for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
                emit(dict(type='frame',timestamp_s=timestamp,frames_processed=count,update_s=time.monotonic()-tick,
                          kv_tokens=live.past_key_values.get_seq_length(),background_response=response))
            ingest_s=time.monotonic()-start;query_start=time.monotonic();kv=live.past_key_values.get_seq_length()
            emit(dict(type='query',timestamp_s=job['t_query'],frames_processed=count,kv_tokens=kv))
            live.inplace_output_ids=torch.zeros(1,128,device='cuda:0',dtype=torch.long)
            _,raw=live._call_for_response(job['t_query'],prompt)
            for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
            response=raw.split('Assistant:',1)[1].strip()
            return dict(response=response,raw_response=raw,output_tokens=len(self.processor.encode(response,add_special_tokens=False)),
                        input_tokens=kv,sampling=frames.meta,frames_processed=count,ingestion_wall_s=ingest_s,
                        query_wall_s=time.monotonic()-query_start,native_api='LiveInfer._call_for_streaming/_call_for_response',
                        interval_threshold=.725,frame_visual_tokens=10,background_generation_cap=100,
                        final_generation_cap=128,kv_tokens_after_answer=live.past_key_values.get_seq_length(),
                        preprocessing='official bicubic long-edge384 + black square pad; causal RGB only')
