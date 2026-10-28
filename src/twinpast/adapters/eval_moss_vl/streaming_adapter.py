# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Drive the official realtime session with acknowledged, ordered inputs."""
import functools,queue,time
from adapter import Adapter as OfflineAdapter
from common import record_inputs
from native import CausalFrames


class Adapter(OfflineAdapter):
    def answer_stream(self,job,prompt,emit):
        import torch
        model=self.model;events=queue.Queue();state=dict(frames=0,query_staged=False,query_processed=False)
        original=model.forward
        @functools.wraps(original)
        def forward(*args,**kwargs):
            output=original(*args,**kwargs)
            if kwargs.get('pixel_values') is not None:
                for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
                state['frames']+=len(kwargs['grid_thw'])
                cache=getattr(output,'past_key_values',None)
                events.put(dict(type='frame_ack',frames_processed=state['frames'],
                     kv_tokens=cache.get_seq_length() if cache is not None else None,**record_inputs(kwargs)))
            if state['query_staged']:
                state['query_processed']=True
                state['query_staged']=False
                events.put(dict(type='query_ack',frames_processed=state['frames']))
            return output
        model.forward=forward
        class PromptQueue(queue.Queue):
            def get(self,*args,**kwargs):
                item=super().get(*args,**kwargs);state['query_staged']=True;return item
        class OutputSink(queue.Queue):
            def put(self,item,*args,**kwargs):
                events.put(dict(type='output',frames_processed=state['frames'],
                                query_processed=state['query_processed'],text=str(item)))
        session=model.create_realtime_session(self.processor,
            initial_prompt='As the video streams frame by frame, describe important changes as they happen. Stay silent when there is no relevant update.',
            frame_queue_size=1,max_tokens_per_turn=86400,do_sample=False,max_new_tokens=4096)
        session._prompt_queue=PromptQueue();session._output_queue=OutputSink()
        frames=CausalFrames(job,fps=1.0);start=time.monotonic();count=0;background_caps=0;latest_kv=None
        def receive(deadline):
            nonlocal latest_kv
            while time.monotonic()<deadline:
                session._raise_if_failed()
                try:event=events.get(timeout=.05)
                except queue.Empty:continue
                emit(event)
                if event['type']=='frame_ack':latest_kv=event['kv_tokens']
                return event
            raise TimeoutError('Native MOSS realtime input/output acknowledgment exceeded 120s')
        try:
            session.start()
            for image,timestamp in frames:
                count+=1;tick=time.monotonic();session.push_frame(image,timestamp=timestamp,drop_oldest=False)
                chunks=0;ack=False
                while True:
                    event=receive(tick+120)
                    if event['type']=='frame_ack' and event['frames_processed']>=count:ack=True
                    if event['type']=='output' and event['frames_processed']>=count:
                        chunks+=1
                        if ack and '<|silence|>' in event['text']:break
                        if ack and chunks>=128:background_caps+=1;break
                emit(dict(type='frame_done',timestamp_s=timestamp,frames_processed=count,update_s=time.monotonic()-tick))
            ingest_s=time.monotonic()-start;query_start=time.monotonic()
            emit(dict(type='query',timestamp_s=job['t_query'],frames_processed=count,kv_tokens=latest_kv))
            session.push_prompt(prompt);chunks=[]
            while True:
                event=receive(query_start+120)
                if event['type']=='output' and event['query_processed']:
                    text=event['text']
                    if '<|silence|>' in text:break
                    if text!='<|round_start|>':chunks.append(text)
                    if len(chunks)>=128:break
            for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
            raw=''.join(chunks);tokenizer=self.processor.tokenizer
            response=tokenizer.decode(tokenizer.encode(raw,add_special_tokens=False),skip_special_tokens=True).strip()
            return dict(response=response,raw_response=raw,output_tokens=len(tokenizer.encode(raw,add_special_tokens=False)),
                sampling=frames.meta,frames_processed=count,ingestion_wall_s=ingest_s,query_wall_s=time.monotonic()-query_start,
                input_tokens=latest_kv,native_api='MossVLRealtimeSession + acknowledged native forward',
                background_chunk_caps=background_caps,pacing='accelerated replay; wait silence or 128 decoded chunks per frame')
        finally:
            session.close(timeout=60);model.forward=original
