import random
import time
import cv2
import platform
import subprocess
import os

from moviepy import AudioFileClip, ImageSequenceClip

from core.plugins.lip_sync.core.avatar import Avatar
from core.plugins.text2speech import Audio


# noinspection PyUnresolvedReferences
def video_writer(avatar, audio_path, output):
    temp = f'_temp_avatar_{time.strftime("%Y%m%d_%H%M%S")}_{random.randint(0, 99999)}.avi'
    generator = avatar.lip_synced_frame_generator(audio_path)
    first_frame_batch = next(generator)
    frame_h, frame_w = first_frame_batch[0].shape[:-1]
    out = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*'DIVX'), self.args.fps, (frame_w, frame_h))
    print(f"write file p1: for {avatar.id}")
    for frame in first_frame_batch:
        out.write(frame)
    for frame_batch in generator:
        print(f"write file p2: for {avatar.id}")
        for frame in frame_batch:
            out.write(frame)
    out.release()
    command = f'ffmpeg -y -i {audio_path} -i {temp} -strict -2 -q:v 1 {output}'
    subprocess.call(command, shell=platform.system() != 'Windows')
    os.remove(temp)


def video_buffer(avatar: Avatar, audio_path, **kwargs):
    avatar.update_args(kwargs)
    audio = Audio.load(audio_path)
    audio_clip = AudioFileClip(audio_path)
    audio_time = avatar.args.buffer_size / avatar.args.fps
    for idx, fr24 in enumerate(avatar.frame_buffer(audio)):
        video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in fr24], fps=24)
        start_time = idx * audio_time
        end_time = (idx + 1) * audio_time
        video_clip = video_clip.with_audio(audio_clip.subclipped(start_time, min(end_time, audio_clip.duration)))
        yield video_clip
