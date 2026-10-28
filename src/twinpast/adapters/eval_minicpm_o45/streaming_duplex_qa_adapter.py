# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Official duplex observation; canonical text chat readout on the same KV.

Bare TEXT units are not chat user turns. The final readout supplies the native
user/assistant template and keeps all duplex KV; no historical pixels replay.
"""
import time
from streaming_duplex_adapter import Adapter as Original
from native import CausalFrames

class Adapter(Original):
    def answer_from_cache(self,prompt,limit=128):
        import torch
        e=self.duplex;tok=e.tokenizer;before=e.decoder.get_cache_length()
        text=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok(text,return_tensors='pt',add_special_tokens=False)['input_ids'].to('cuda:0');size=ids.shape[1]
        stop={tok.eos_token_id}
        # Chunk EOS terminates a duplex unit, not an entire QA answer. Let the
        # text generation continue through it; only logical turn EOS stops QA.
        for token in ['<|im_end|>','<|turn_eos|>']:
            if token in tok.get_vocab():stop.add(tok.convert_tokens_to_ids(token))
        output=self.model.llm.generate(input_ids=ids,past_key_values=e.decoder.cache,
            cache_position=torch.arange(before,before+size,device=ids.device),
            attention_mask=torch.ones((1,before+size),device=ids.device,dtype=torch.long),
            max_new_tokens=limit,do_sample=False,eos_token_id=sorted(x for x in stop if x is not None),pad_token_id=tok.eos_token_id)
        generated=output[0,size:];raw=tok.decode(generated,skip_special_tokens=False)
        return dict(response=tok.decode(generated,skip_special_tokens=True).strip(),raw_response=raw,
            raw_generated_token_ids=generated.cpu().tolist(),output_tokens=len(generated),
            query_template=text,kv_before_query=before,final_kv_tokens=e.decoder.get_cache_length())

    def answer_stream(self,job,prompt,emit):
        import torch
        e=self.duplex;frames=CausalFrames(job,fps=1.0);background=[];count=0;begin=time.monotonic()
        with torch.inference_mode():
            e.prepare(prefix_system_prompt='Streaming Omni Conversation.');initial=e.decoder.get_cache_length()
            for image,timestamp in frames:
                prefill=e.streaming_prefill(frame_list=[image],max_slice_nums=1,batch_vision_feed=False)
                if not prefill['success']:raise RuntimeError('Official duplex prefill failed: '+str(prefill))
                result=e.streaming_generate(max_new_speak_tokens_per_chunk=20,decode_mode='sampling');count+=1
                event=dict(type='duplex_unit',timestamp_s=timestamp,frames_received=count,
                    is_listen=result['is_listen'],response=result['text'],end_of_turn=result['end_of_turn'],
                    output_tokens=result['n_tokens'],kv_tokens=e.decoder.get_cache_length())
                emit(event)
                if result['text']:background.append(event)
            for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
            ingestion=time.monotonic()-begin;start=time.monotonic()
            emit(dict(type='query',timestamp_s=job['t_query'],frames_received=count,kv_tokens=e.decoder.get_cache_length(),readout='native chat user/assistant turn on retained duplex KV'))
            result=self.answer_from_cache(prompt)
            for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
            emit(dict(type='query_result',**result))
        return dict(**result,input_tokens=result['kv_before_query'],ingestion_wall_s=ingestion,
            query_wall_s=time.monotonic()-start,frames_received=count,frames_processed=count,sampling=frames.meta,
            native_api='official MiniCPMODuplex observation + native chat-template QA on identical retained KV',
            initial_kv_tokens=initial,duplex_generate_calls=count,background_responses=background,
            observation_sampling=dict(temperature=.7,top_k=100,top_p=.8,chunk_tokens=20),
            query_sampling='greedy max128; stop at logical turn/im_end/tokenizer EOS, not chunk boundary',
            query_readout_protocol='text QA extension; not the bare TEXT unit condition',
            history_reencoded=False,cache_reset_before_query=False,sliding_window_mode='off (official duplex default)')
