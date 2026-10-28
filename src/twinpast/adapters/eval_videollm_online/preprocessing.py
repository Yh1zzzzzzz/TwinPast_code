# SPDX-License-Identifier: Apache-2.0
# Adapted for the TwinPast public data interface.
"""Released VideoLLM scale-long-edge/black-pad geometry, without video recoding."""
import subprocess


def resize_and_pad(frames, resolution=384):
    import numpy as np
    import torch
    pixels=np.ascontiguousarray(frames,dtype=np.uint8)
    if pixels.ndim==3:pixels=pixels[None]
    count,height,width,channels=pixels.shape
    assert channels==3 and count>0
    # This is the official data/utils.py ffmpeg_once scale/pad expression.
    # Operate only on already-selected causal RGB frames. No temporal filter,
    # extra source frames, or lossy output encoding is introduced.
    filt=(f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':"
          f"'if(gt(iw\\,ih)\\,-2\\,{resolution})',"
          f"pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color=black")
    result=subprocess.run([
        'ffmpeg','-nostdin','-loglevel','error','-sws_flags','bicubic',
        '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
        '-framerate','2','-i','pipe:0','-an','-threads','1','-vf',filt,
        '-frames:v',str(count),'-f','rawvideo','-pix_fmt','rgb24','pipe:1',
    ],input=pixels.tobytes(),capture_output=True,check=True,timeout=60)
    assert len(result.stdout)==count*resolution*resolution*3
    output=np.frombuffer(result.stdout,dtype=np.uint8).reshape(count,resolution,resolution,3).copy()
    return torch.from_numpy(output).permute(0,3,1,2).float()
