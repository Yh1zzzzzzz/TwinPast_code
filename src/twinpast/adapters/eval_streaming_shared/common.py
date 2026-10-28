# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Causal offline VQA sampling; answer labels are not model inputs."""
from release_paths import ROOT as RELEASE_ROOT, ADAPTERS, MODEL_ROOT, RUN_ROOT
import json, math, os, fcntl
from pathlib import Path
import numpy as np
ROOT=RELEASE_ROOT
CACHE_ROOT=RUN_ROOT / 'adapter_cache'
PIXELS=151200

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)); tmp.replace(path)

def sample(job):
    from decord import VideoReader, cpu
    from PIL import Image
    path=Path(job['video']); st=path.stat()
    fingerprint={'size':st.st_size,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns}
    assert fingerprint==job['video_stat'], ('Video changed',str(path))
    base=CACHE_ROOT/'artifacts/frame_cache'/job['job_id']; base.parent.mkdir(parents=True,exist_ok=True)
    with open(str(base)+'.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        meta_path=Path(str(base)+'.json'); array_path=Path(str(base)+'.npy')
        if not (meta_path.exists() and array_path.exists()):
            vr=VideoReader(str(path),ctx=cpu(0),num_threads=1)
            # Never admit a frame whose presentation starts after the query.
            all_ts=np.asarray(vr.get_frame_timestamp(np.arange(len(vr))))[:,0]
            eligible=np.flatnonzero(all_ts<=job['t_query']+1e-6)
            if not len(eligible): raise ValueError('No causal video frames')
            ids=eligible[np.linspace(0,len(eligible)-1,min(64,len(eligible)),dtype=int)]
            raw=vr.get_batch(ids).asnumpy(); frames=[]
            for a in raw:
                im=Image.fromarray(a); w,h=im.size
                scale=min(1,math.sqrt(PIXELS/(w*h)))
                if scale<1: im=im.resize((max(1,int(w*scale)),max(1,int(h*scale))),Image.Resampling.LANCZOS)
                frames.append(np.asarray(im))
            frames=np.stack(frames)
            meta={'source':str(path),'fingerprint':fingerprint,'indices':ids.tolist(),
                  'timestamps_s':all_ts[ids].tolist(),'source_frames':len(vr),'source_fps':vr.get_avg_fps(),
                  'source_duration_s':float(vr.get_frame_timestamp(len(vr)-1)[1]),'shape':list(frames.shape),
                  'sampling':'uniform64_causal_prefix','pixel_cap':PIXELS}
            with open(str(array_path)+'.tmp','wb') as f: np.save(f,frames)
            os.replace(str(array_path)+'.tmp',array_path); dump(meta_path,meta)
        meta=json.loads(meta_path.read_text()); assert meta['fingerprint']==fingerprint
        return np.load(array_path,allow_pickle=False),meta

def prompt(job):
    return job['question']+'\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(job['options']))+'\nAnswer with only the letter of the correct option.'

def metadata(meta):
    from transformers.video_utils import VideoMetadata
    return VideoMetadata(total_num_frames=meta['source_frames'],fps=meta['source_fps'],
       duration=meta['source_duration_s'],height=meta['shape'][1],width=meta['shape'][2],
       frames_indices=meta['indices'],video_backend='decord')

def fast_patch(model):
    """Equivalent GEMM only for non-overlapping one-output patch Conv3d."""
    import torch, types
    for mod in model.modules():
        proj=getattr(mod,'proj',None)
        if not isinstance(proj,torch.nn.Conv3d): continue
        if proj.groups!=1 or proj.kernel_size!=proj.stride or any(proj.padding): continue
        # Preserve all operations of known Qwen-style patch embed only.
        if mod.__class__.__name__ not in {'Qwen3VLVisionPatchEmbed','MossVLVisionPatchEmbed','Qwen2_5_VisionPatchEmbed'}: continue
        def forward(self,x):
            p=self.proj
            return torch.nn.functional.linear(x.reshape(-1,p.weight[0].numel()).to(p.weight.dtype),p.weight.flatten(1),p.bias)
        mod.forward=types.MethodType(forward,mod)

def load_kwargs():
    import torch
    return dict(trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,
                device_map='auto',max_memory={i:'20GiB' for i in range(torch.cuda.device_count())},attn_implementation='sdpa')

def record_inputs(inputs):
    return {'input_tokens':int(inputs['input_ids'].shape[-1]) if 'input_ids' in inputs else None,
            'input_shapes':{k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')},
            'grids':{k:v.detach().cpu().tolist() for k,v in inputs.items() if k.endswith('grid_thw') or k=='grid_thw'}}
