# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Released incremental clip scheduler; explicit extension for a held-out question."""
from dataclasses import asdict
import time
from adapter import Adapter as OfflineAdapter
from native import CausalFrames


class Adapter(OfflineAdapter):
    def __init__(self):
        super().__init__()
        from dispider.streaming.dispider_adapter import ReleasedPerceptionDecisionReactionBackend
        r=self.runtime
        self.backend=ReleasedPerceptionDecisionReactionBackend(self.model,r.tokenizer,r.image_processor,r.time_tokenizer,
                                                               generation_kwargs=dict(max_new_tokens=128,do_sample=False))

    def answer_stream(self,job,prompt,emit):
        import torch
        from dispider.streaming.dispider_adapter import DispiderStreamingAdapter
        backend=self.backend;backend.reset()
        adapter=DispiderStreamingAdapter(backend,
             'Observe the video and describe important events for a later question.',enable_decision_kv_cache=True)
        session=adapter.new_session(window_seconds=16,frames_per_window=16,max_history_windows=8)
        # 2fps contains every 0.5,1.5,... midpoint of the native 1fps windows;
        # boundaries trigger scheduling. The last causal source frame matches
        # native partial-window nearest sampling without decoding all CFR24.
        frames=CausalFrames(job,fps=2.,include_query_frame=True);start=time.monotonic();count=0
        last_clips=0
        try:
            with torch.inference_mode():
                for image,timestamp in frames:
                    count+=1;tick=time.monotonic();events=session.push_frames([image],[timestamp])
                    for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
                    if len(adapter.clips)!=last_clips:
                        last_clips=len(adapter.clips)
                        emit(dict(type='window',timestamp_s=timestamp,retained_clips=last_clips,
                           scheduler_history_windows=len(session.history),update_s=time.monotonic()-tick,
                           decision=asdict(adapter.decisions[-1]),responses=[e.as_dict() for e in events]))
                events=session.flush()
                for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
                emit(dict(type='flush',timestamp_s=job['t_query'],retained_clips=len(adapter.clips),
                    responses=[e.as_dict() for e in events],decisions=[asdict(x) for x in adapter.decisions]))
                ingest_s=time.monotonic()-start;query_start=time.monotonic()
                emit(dict(type='query',timestamp_s=job['t_query'],frames_received=count,
                          retained_clips=len(adapter.clips),stable_triggers=list(adapter.committed_trigger_positions)))
                # Native cached Decision prefixes are question-dependent. Clear
                # only that cache; retain the native clip state and decisions.
                # Reaction uses the released backend's normal memory-read path.
                backend.reset()
                text=backend.react(adapter.clips,question=prompt,committed_trigger_positions=adapter.committed_trigger_positions)
                for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
                return dict(response=text,output_tokens=len(self.runtime.tokenizer.encode(text,add_special_tokens=False)),
                    sampling=frames.meta,frames_received=count,frames_processed=16*len(adapter.clips),
                    ingestion_wall_s=ingest_s,query_wall_s=time.monotonic()-query_start,input_tokens=None,
                    native_api='StreamingSession + cached DispiderStreamingAdapter + query-time Reaction',
                    question_protocol='generic observation during stream; held-out question at Query; Decision KV cleared',
                    native_backend_retained_clips=len(adapter.clips),scheduler_history_windows=len(session.history),
                    decision_cache_enabled=True,stable_trigger_positions=list(adapter.committed_trigger_positions))
        finally:adapter.reset()
