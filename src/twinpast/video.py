"""Causal media access for independently configured model adapters."""
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

from .data import require


def check_identity(job):
    st = Path(job["video"]).stat()
    require({"size": st.st_size, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns} == job["video_stat"],
            "Video changed after input preparation")


def complete_video(job):
    """Whole-video APIs require a file whose last frame is within the query prefix."""
    check_identity(job)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=codec_type,nb_frames,avg_frame_rate,r_frame_rate", "-of", "json", job["video"]],
                           capture_output=True, text=True, check=True, timeout=30)
    streams = json.loads(probe.stdout)["streams"]
    videos = [s for s in streams if s["codec_type"] == "video"]
    require(len(videos) == 1 and all(s["codec_type"] == "video" for s in streams),
            "Whole-video input requires one video track and no audio")
    stream = videos[0]
    rate = Fraction(stream["avg_frame_rate"])
    require(rate > 0 and rate == Fraction(stream["r_frame_rate"]), "Whole-video input requires constant frame rate")
    require((int(stream["nb_frames"]) - 1) / float(rate) <= job["t_query"] + 1e-6,
            "Video contains post-query frames; supply an explicitly trimmed prefix")
    return Path(job["video"])


def sampled_images(job, mode="uniform64", pixel_cap=151200):
    import numpy as np
    from decord import VideoReader, cpu
    from PIL import Image
    check_identity(job)
    reader = VideoReader(job["video"], ctx=cpu(0), num_threads=1)
    times = np.asarray(reader.get_frame_timestamp(np.arange(len(reader))))[:, 0]
    eligible = np.flatnonzero(times <= job["t_query"] + 1e-6)
    require(len(eligible) > 0, "No causal video frames")
    if mode == "uniform64":
        indices = eligible[np.linspace(0, len(eligible) - 1, min(64, len(eligible)), dtype=int)]
    elif mode == "latest":
        indices = eligible[-1:]
    elif mode == "recent4":
        targets = np.linspace(max(0, job["t_query"] - 4), job["t_query"], 4)
        requested = np.floor(targets * reader.get_avg_fps() + 1e-8).astype(int)
        indices = np.minimum(requested, eligible[-1])
    else:
        raise ValueError("Unknown sampling mode")
    images = []
    for array in reader.get_batch(indices).asnumpy():
        image = Image.fromarray(array)
        width, height = image.size
        scale = min(1.0, math.sqrt(pixel_cap / (width * height)))
        if scale < 1:
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
        images.append(image)
    return images, {"indices": indices.tolist(), "timestamps_s": times[indices].tolist(),
                    "sampling": mode, "pixel_cap": pixel_cap}

