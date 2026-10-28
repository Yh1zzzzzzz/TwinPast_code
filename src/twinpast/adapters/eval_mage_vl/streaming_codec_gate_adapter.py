# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Official codec/StreamMind gate with a held-out final benchmark question.

The released demo gates causal segment prefixes and generates from the current
segment only. No caption history or discarded visual segment is added to the
final QA.
"""
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
import os
import subprocess
import time
from pathlib import Path

from common import ROOT, load_kwargs, record_inputs

CACHE_ROOT = RUN_ROOT / 'adapter_cache'
GENERIC = 'Please describe the video content in detail based on the provided information.'


def segment_bounds(total_frames, fps=24, seconds=8):
    width = fps * seconds
    return [(lo, min(lo + width, total_frames)) for lo in range(0, total_frames, width)]


def is_unencodable_tail(error, lo, hi, total):
    # Match the upstream skip only for an incomplete final group producing
    # no canvases. Propagate failures in complete segments or other errors.
    return (hi == total and 0 < hi - lo < 32
            and 'no canvases produced' in str(error))


def extract_exact_segment(source, target, lo, hi):
    """Make an evaluation cache; preserve source pixels and exact causal bounds.

    Demo -c copy may include neighboring GOP frames. Lossless H.264 avoids that
    tolerance here; codec bit costs belong to this derivative, not the source.
    """
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name('.tmp_' + target.name)
    subprocess.run([
        'ffmpeg', '-nostdin', '-y', '-loglevel', 'error', '-threads', '2',
        '-i', str(source), '-vf', f'trim=start_frame={lo}:end_frame={hi},setpts=PTS-STARTPTS',
        '-an', '-c:v', 'libx264', '-qp', '0', '-preset', 'ultrafast',
        '-threads', '2', '-g', '192', '-vsync', '0', '-frames:v', str(hi - lo), str(tmp),
    ], capture_output=True, check=True, timeout=180)
    tmp.replace(target)


class Adapter:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        path = str(MODEL_ROOT / 'Mage-VL')
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs()).eval()
        # Load the actual released checkpoint, strict=True in the official loader.
        self.gate = self.model.model._load_streammind_gate()
        self.visual_device = next(self.model.model.visual.parameters()).device
        self.text_device = self.model.get_input_embeddings().weight.device
        os.environ['ONLINE_CODEC_CACHE_DIR'] = str(CACHE_ROOT / 'artifacts/mage_streaming_codec_cache')

    def make_inputs(self, clip, prompt):
        p = self.processor
        text = p.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'video'}, {'type': 'text', 'text': prompt},
        ]}], tokenize=False, add_generation_prompt=True)
        return p(text=[text], videos=[str(clip)], video_backend='codec',
                 codec_config={'patch': 16, 'max_pixels': 150000},
                 max_pixels=150000, return_tensors='pt', padding=False)

    def generate(self, inputs, limit):
        import torch
        kwargs = {k: v.to(self.text_device, dtype=self.model.dtype) if k == 'pixel_values'
                  else v.to(self.text_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**kwargs, max_new_tokens=limit, do_sample=False)
        ids = output[0, kwargs['input_ids'].shape[1]:]
        return self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip(), len(ids)

    def answer_stream(self, job, prompt, emit):
        import math
        import torch
        from decord import VideoReader, cpu

        source = Path(job['video']); st = source.stat()
        assert dict(size=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns) == job['video_stat']
        reader = VideoReader(str(source), ctx=cpu(0), num_threads=1)
        assert abs(reader.get_avg_fps() - 24) < 1e-3
        total = min(len(reader), int(math.floor(job['t_query'] * 24 + 1e-6)) + 1)
        del reader
        assert total > 0 and (total - 1) / 24 <= job['t_query'] + 1e-6
        tokens, lengths, background, unencodable = [], [], [], []
        begin = time.monotonic(); last_clip = None; last_encoded_hi = 0
        for index, (lo, hi) in enumerate(segment_bounds(total)):
            clip = CACHE_ROOT / 'artifacts/mage_streaming_clips' / job['job_id'] / f'{lo:08d}_{hi:08d}.mp4'
            extract_exact_segment(source, clip, lo, hi)
            try:
                inputs = self.make_inputs(clip, GENERIC)
            except RuntimeError as error:
                if not is_unencodable_tail(error, lo, hi, total):
                    raise
                event = dict(type='codec_unencodable_tail', timestamp_s=(hi - 1) / 24,
                    source_indices=[lo, hi - 1], source_frames=hi - lo,
                    reason='official codec produced no canvases for final incomplete group',
                    official_behavior='inference_streaming.py skips processor-unusable segments',
                    fallback_backend=False, error=str(error))
                unencodable.append(event);emit(event)
                continue
            visual = {k: inputs[k].to(self.visual_device, dtype=self.model.dtype) if k == 'pixel_values'
                      else inputs[k].to(self.visual_device)
                      for k in ['pixel_values', 'image_grid_thw', 'patch_positions'] if k in inputs}
            with torch.inference_mode():
                # Encode each arriving segment once. The released gate consumes
                # all retained causal feature tokens; no future prefix is used.
                token = self.model.model._streammind_vision_tokens(**visual)
                tokens.append(token); lengths.append(token.shape[1])
                boundaries = torch.tensor(lengths, device=self.visual_device).cumsum(0)
                logits = self.gate(torch.cat(tokens, dim=1), response_positions=boundaries)[0]
                probability = float(logits[-1].float().softmax(-1)[1])
            response = None
            if probability >= .5:
                response, output_tokens = self.generate(inputs, 80)
                background.append(dict(segment=index, response=response, output_tokens=output_tokens))
            emit(dict(type='segment', timestamp_s=(hi - 1) / 24, source_indices=[lo, hi - 1],
                      codec_backend='hevc API on lossless H264 derivative', gate_probability=probability,
                      gate_threshold=.5, gate_open=probability >= .5, background_response=response,
                      cumulative_gate_steps=sum(lengths), retained_decoder_segments=1,
                      retained_gate_segments=len(tokens), input_shapes=record_inputs(inputs)['input_shapes']))
            last_clip = clip; last_encoded_hi = hi
            del inputs, visual
        if last_clip is None:
            raise RuntimeError('No usable causal codec segments; cannot answer this session')
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        ingestion = time.monotonic() - begin
        start = time.monotonic()
        emit(dict(type='query', timestamp_s=job['t_query'], decoder_context='last current segment only',
                  forced_user_query=True, gate_controls_proactive_responses_only=True))
        # The paper allows user queries independent of proactive triggering.
        # Keep the release's current-segment decoder context; no caption memory.
        inputs = self.make_inputs(last_clip, prompt)
        response, output_tokens = self.generate(inputs, 128)
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        return dict(response=response, output_tokens=output_tokens, ingestion_wall_s=ingestion,
                    query_wall_s=time.monotonic() - start, gate_exercised=True,
                    native_api='official codec segments + StreamMind gate; held-out query on current segment',
                    segment_seconds=8, gate_threshold=.5, retained_decoder_segments=1,
                    background_responses=background, gate_segments=len(tokens),
                    unencodable_tail_segments=unencodable,
                    encoded_source_frames=total-sum(e['source_frames'] for e in unencodable),
                    last_encoded_source_timestamp=(last_encoded_hi-1)/24,
                    codec_config={'patch':16, 'max_pixels':150000, 'target_canvas':'checkpoint default 32'},
                    sampling={'source_frames':total, 'source_fps':24, 'last_source_timestamp':(total-1)/24,
                              'exact_lossless_segment_derivatives':True, 'source_unchanged':True},
                    **record_inputs(inputs))
