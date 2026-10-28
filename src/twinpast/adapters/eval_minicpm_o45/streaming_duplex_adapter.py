# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Released MiniCPMODuplex vision/text listen-speak engine, question held out.

Vision-only data has no audio targets. Skip only Token2wav asset initialization
(the official wrapper calls it unconditionally even with generate_audio=False).
No decoder, cache, listen/speak or sampling method is replaced.
"""
import time
from adapter import Adapter as Base
from native import CausalFrames

class Adapter(Base):
    def __init__(self):
        super().__init__()
        original=self.model.init_tts
        self.model.init_tts=lambda **kwargs:None
        try:self.duplex=self.model.as_duplex(device='cuda:0',generate_audio=False)
        finally:self.model.init_tts=original

    def answer_stream(self,job,prompt,emit):
        import torch
        frames=CausalFrames(job,fps=1.0)
        engine=self.duplex;count=0;calls=0;background=[]
        begin=time.monotonic()
        with torch.inference_mode():
            engine.prepare(prefix_system_prompt='Streaming Omni Conversation.')
            initial=engine.decoder.get_cache_length()
            for image,timestamp in frames:
                prefill=engine.streaming_prefill(frame_list=[image],max_slice_nums=1,batch_vision_feed=False)
                if not prefill['success']:raise RuntimeError('Official duplex prefill failed: '+str(prefill))
                result=engine.streaming_generate(max_new_speak_tokens_per_chunk=20,decode_mode='sampling')
                count+=1;calls+=1
                event=dict(type='duplex_unit',timestamp_s=timestamp,frames_received=count,
                    is_listen=result['is_listen'],response=result['text'],end_of_turn=result['end_of_turn'],
                    output_tokens=result['n_tokens'],kv_tokens=engine.decoder.get_cache_length())
                emit(event)
                if result['text']:background.append(event)
            for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
            ingestion=time.monotonic()-begin;start=time.monotonic()
            before=engine.decoder.get_cache_length()
            emit(dict(type='query',timestamp_s=job['t_query'],frames_received=count,kv_tokens=before))
            prefill=engine.streaming_prefill(text_list=[prompt])
            if not prefill['success']:raise RuntimeError('Official duplex query prefill failed: '+str(prefill))
            generated_start=len(engine.total_ids)
            result=engine.streaming_generate(max_new_speak_tokens_per_chunk=128,decode_mode='sampling')
            raw_generated_ids=engine.total_ids[generated_start:]
            raw_generated_text=engine.tokenizer.decode(raw_generated_ids,skip_special_tokens=True)
            calls+=1
            for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
            final=engine.decoder.get_cache_length()
            emit(dict(type='duplex_query_result',is_listen=result['is_listen'],response=result['text'],
                end_of_turn=result['end_of_turn'],output_tokens=result['n_tokens'],kv_tokens=final))
        return dict(response=result['text'],official_result_text=result['text'],
            raw_generated_token_ids=raw_generated_ids,raw_generated_text=raw_generated_text,
            output_tokens=result['n_tokens'],input_tokens=before,
            ingestion_wall_s=ingestion,query_wall_s=time.monotonic()-start,
            frames_received=count,frames_processed=count,sampling=frames.meta,
            native_api='official MiniCPMODuplex prepare + per-second prefill/generate + held-out TEXT query',
            duplex_generate_calls=calls,initial_kv_tokens=initial,final_kv_tokens=final,
            query_is_listen=result['is_listen'],background_responses=background,
            sliding_window_mode='off (official duplex default)',
            audio_adaptation='generate_audio=False; Token2wav initializer skipped; no audio input or synthesis',
            generation_parameters=dict(temperature=.7,top_k=100,top_p=.8,repetition_penalty=1.05,
                observation_chunk_tokens=20,final_chunk_tokens=128,force_listen_count=0))
