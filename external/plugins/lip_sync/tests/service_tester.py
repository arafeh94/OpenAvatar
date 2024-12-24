import asyncio
import os
import tempfile
from fastapi import FastAPI
from moviepy import ImageSequenceClip

from external.plugins.lip_sync.core.avatar import Avatar
from external.plugins.lip_sync.core.models import AvatarWave2LipModel

app = FastAPI()
model = AvatarWave2LipModel()


def get_buffer():
    audio_path = "/home/arafeh/PycharmProjects/avatar_rag_back_end/avatar/avatar_only/avatar_dir/harvard.wav"
    avatar = Avatar("lisa_casual_720_pl", model)
    avatar.init()
    return avatar.video_buffer(audio_path)


clip_buffer = get_buffer()
iteration = 0


def stream_clip(clip: ImageSequenceClip):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.close()
        clip.write_videofile(tmpfile.name, codec="libx264")
        yield open(tmpfile.name, mode="rb").read()
        os.remove(tmpfile.name)


# @app.get("/")
async def avatar():
    global clip_buffer, iteration
    print(f"generating new clip, iteration: {iteration}")
    try:
        iteration += 1
        clip = next(clip_buffer)
        clip.preview()
    except StopIteration:
        iteration = 0
        clip_buffer = get_buffer()
        clip = next(clip_buffer)
        clip.preview()


if __name__ == "__main__":
    asyncio.run([avatar()])
