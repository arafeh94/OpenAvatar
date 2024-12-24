import asyncio
import os
import tempfile
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from moviepy import ImageSequenceClip

from external.core.utils.lazy_loader import LazyLoader
from external.core.utils.token_generator import generate_token
from external.plugins.lip_sync.core.avatar import Avatar
from external.plugins.lip_sync.core.models import AvatarWave2LipModel
from external.plugins.text2speech import Text2Speech

app = FastAPI()
avatar_model = AvatarWave2LipModel()
speech_loader = LazyLoader(Text2Speech)
buffers = {}


def register_buffer(audio_path, avatar_cache="lisa_casual_720_pl"):
    token = generate_token()
    avatar = Avatar(avatar_cache, avatar_model)
    avatar.init()
    buffers[token] = avatar.video_buffer(audio_path)
    return token


def get_next_clip(token):
    if token not in buffers:
        raise KeyError(f"Request token [{token}] does not exist. Should register first")
    return next(buffers[token], None)


def get_buffer():
    audio_path = "/home/arafeh/PycharmProjects/avatar_rag_back_end/avatar/avatar_only/avatar_dir/harvard.wav"
    avatar = Avatar("lisa_casual_720_pl", avatar_model)
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


@app.get("/")
async def avatar():
    global clip_buffer, iteration
    print(f"generating new clip, iteration: {iteration}")
    try:
        iteration += 1
        clip = next(clip_buffer)
        return StreamingResponse(stream_clip(clip), media_type="video/mp4")
    except StopIteration:
        iteration = 0
        clip_buffer = get_buffer()
        clip = next(clip_buffer)
        return StreamingResponse(stream_clip(clip), media_type="video/mp4")


@app.get("/request")
async def request_avatar(text: str, voice_id: int) -> dict:
    speech_loader.get().convert(text, voice_id).as_file("avatar.wav")
    token = register_buffer("avatar.wav")
    return {'token': token}


async def run_main():
    text = ("hello samira, this is a very good night, I would like to tell you something."
            "This night is very beautiful, to the point, that I would like to show you "
            "how much I love you. I love soo much, to the point of no return. bye bye.")
    token_request = await request_avatar(text, 2)
    token = token_request['token']
    while True:
        clip = get_next_clip(token)
        if clip is None:
            break
        clip.preview()


# uvicorn.run(app, host="localhost", port=8000)
if __name__ == "__main__":
    asyncio.run(run_main())
    print("samira")
