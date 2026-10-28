# SPDX-License-Identifier: Apache-2.0
import fcntl, math, subprocess, time
from .config import RUN_ROOT, Path, read, dump, stat, check
from .official import upstream_bench, SILENT

def prepare(job,condition):
    folder=RUN_ROOT/'cache'/condition/job['job_id'];folder.mkdir(parents=True,exist_ok=True)
    with (folder/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _prepare(job,condition)

def _prepare(job,condition):
    check(job)
    folder=RUN_ROOT/'cache'/condition/job['job_id']
    saved=folder/'prepared.json'
    if saved.exists():
        data=read(saved)
        assert data['source_stat']==job['video_stat']
        assert all(stat(s['path'])==s['stat'] for s in data['segments'])
        return data
    from decord import VideoReader,cpu
    vr=VideoReader(job['video'],ctx=cpu(0),num_threads=1)
    fps=float(vr.get_avg_fps());total=len(vr);last=float(vr.get_frame_timestamp(total-1)[0])
    assert last<=job['t_query']+1e-4,(job['job_id'],'future frames',last,job['t_query'])
    start=time.monotonic()
    attempt=folder/('attempt_'+str(time.time_ns()));attempt.mkdir(parents=True,exist_ok=False)
    if condition=='tail30':
        paths=upstream_bench()['split_video_from_end'](job['video'],str(attempt),30)
        duration=upstream_bench()['get_video_duration'](job['video'])
        offset=round(round(duration,2)-len(paths),2) if duration>=1 else 0
        segments=[dict(path=str(Path(p).resolve()),stat=stat(p),index=i,source_interval=[offset+i,min(duration,offset+i+1)]) for i,p in enumerate(paths)]
        sampling='Published end-aligned one-second split_video_from_end; fps1/min2 temporal frames.'
    else:
        # Current release is CFR and its final native PTS is exactly the query.
        assert abs(fps-round(fps))<1e-6 and int(round(fps))%2==0,fps
        fps=int(round(fps));stride=fps//2
        cmd=['ffmpeg','-nostdin','-v','error','-threads','2','-i',job['video'],'-an','-sn','-filter_threads','1',
             '-vf',"select='not(mod(n,{}))',setpts=N/(2*TB)".format(stride),'-r','2','-c:v','libx264',
             '-preset','fast','-crf','0','-pix_fmt','yuv420p','-threads','2','-bf','0','-g','2','-keyint_min','2',
             '-sc_threshold','0','-f','segment','-segment_time','1','-reset_timestamps','1',str(attempt/'chunk_%05d.mp4')]
        subprocess.run(cmd,check=True,capture_output=True)
        paths=sorted(attempt.glob('chunk_*.mp4'))
        assert len(paths)==math.ceil(total/fps),(len(paths),total,fps)
        segments=[]
        for i,p in enumerate(paths):
            ids=list(range(i*fps,min(total,(i+1)*fps),stride))
            times=[n/fps for n in ids]
            assert times and max(times)<=job['t_query']+1e-6
            segments.append(dict(path=str(p.resolve()),stat=stat(p),index=i,source_frame_indices=ids,
                source_timestamps=times,source_visible_last_frame_time=(min(total,(i+1)*fps)-1)/fps))
        sampling='Origin anchored2FPS, lossless H264 chunks; temporal patch duplicate adds no evidence.'
    data=dict(source=job['video'],source_stat=job['video_stat'],query=job['t_query'],native_fps=fps,native_frames=total,
              source_last_pts=last,segments=segments,sampling=sampling,prepare_seconds=time.monotonic()-start,
              saved_bytes=sum(s['stat']['size'] for s in segments))
    dump(saved,data)
    return data

def benchmark_messages(media, prompt):
    up = upstream_bench()
    messages = [{'role':'system', 'content':up['SYSTEM_PROMPT']}]
    for i, segment in enumerate(media['segments']):
        content = [{'type':'video', 'video':segment['path']}]
        if i == len(media['segments']) - 1:
            content.append({'type':'text','text':prompt})
        messages.append({'role':'user','content':content})
        if i < len(media['segments']) - 1:
            messages.append({'role':'assistant','content':SILENT})
    return messages

def portable_messages(messages):
    """Normalize upstream history video tuples to a local video path."""
    result = []
    for message in messages:
        content = message['content']
        if isinstance(content, list):
            content = [dict(x) for x in content]
            for item in content:
                if item.get('type') == 'video' and isinstance(item['video'], (tuple, list)):
                    # StreamingSession stores a singleton tuple; upstream also
                    # accepts decoded tensor/metadata tuples in its demo.
                    if len(item['video']) != 1 or not isinstance(item['video'][0], str):
                        raise ValueError('This backend requires a local video path tuple')
                    item['video'] = item['video'][0]
        result.append({'role':message['role'], 'content':content})
    return result

