# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Official OneStreamer PHCM caption pass and recent16 memory QA wrapper."""
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
import json
from pathlib import Path
from types import SimpleNamespace

class Adapter:
    def __init__(self):
        from onestreamer.engine import OneStreamerEngine
        root=Path(__file__).parent
        self.caption=json.loads((root/'upstream/Eval/Proactive_Eval/OneStreamer/configs/streamingbench.json').read_text())['caption']
        self.engine=OneStreamerEngine(model_path=str(MODEL_ROOT / 'OneStreamer-4B'),dtype='bfloat16',attn_implementation='sdpa',min_pixels=3136,max_pixels=100352)
        self.model=self.engine.model
        from common import fast_patch
        fast_patch(self.model)
        # The released checkpoint attaches its template to the processor only;
        # official caption code calls the tokenizer. Use the same supplied text.
        if not self.engine.processor.tokenizer.chat_template:
            template=json.loads(Path(str(MODEL_ROOT / 'OneStreamer-4B/chat_template.json')).read_text())['chat_template']
            self.engine.processor.tokenizer.chat_template=template

    def answer_stream(self,job,question,emit):
        from onestreamer.caption_video import VideoCaptionSession,iter_video_updates
        from onestreamer.caption_optimized import SegmentScopeTracker,finalize_caption_boundary
        from onestreamer.memory import MemoryBank,MemoryEvent
        from onestreamer.prompts import MEMORY_QA_SYSTEM
        from onestreamer.video import read_recent_frames
        # Validate current input identity before invoking upstream decoders.
        stat=Path(job['video']).stat();assert dict(size=stat.st_size,inode=stat.st_ino,mtime_ns=stat.st_mtime_ns)==job['video_stat']
        sample=SimpleNamespace(sample_id=job['job_id'],start=0.,end=job['t_query'])
        # No question, options, answer or evidence annotations enter caption construction.
        session=VideoCaptionSession(self.engine,sample,self.caption)
        tracker=SegmentScopeTracker(0.,control_protocol=self.caption['control_protocol'])
        events=[];outputs=[];count=0
        for idx,(start,end,update) in enumerate(iter_video_updates(job['video'],0.,job['t_query'],4.)):
            assert all(0<=t<=job['t_query'] for t in update.timestamps)
            raw=session.step(update,idx);event=tracker.parse_online(raw,start,end)
            record=dict(round=idx,start=start,end=end,kind=event.kind,memory_start=event.start,memory_end=event.end,output=raw)
            outputs.append(record);count+=len(update.frames)
            if event.kind in {'response','summary'} and event.text:events.append(event)
            emit(dict(type='caption',timestamp_s=end,query_seen=False,timestamps_s=update.timestamps,**record))
        boundary=finalize_caption_boundary(engine=self.engine,session=session,causal_events=events,causal_outputs=outputs,segment_start=tracker.segment_start,start=0.,end=job['t_query'],caption_config=self.caption)
        emit(dict(type='caption_boundary',**boundary))
        bank=MemoryBank(MemoryEvent.from_dict(e) for e in boundary['events'])
        text,meta=bank.render_with_budget('hierarchical',job['question'],self.engine.processor.tokenizer,12000)
        recent=read_recent_frames(job['video'],0.,job['t_query'],16,1.)
        assert all(t<=job['t_query'] for t,_ in recent)
        response=self.engine.answer(MEMORY_QA_SYSTEM,text,recent,question,max_tokens=16,recent_resize_policy='source_pixels')
        emit(dict(type='answer',response=response,memory_text=text,memory_meta=meta,recent_timestamps=[t for t,_ in recent],**self.engine.last_input_metrics))
        return dict(response=response,frames=count,memory_meta=meta,recent_timestamps=[t for t,_ in recent],readout='free_generation',generation_limit=16,**self.engine.last_input_metrics)
