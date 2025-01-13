import os
import tempfile
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from moviepy import ImageSequenceClip
from starlette.middleware.cors import CORSMiddleware

from external.core.utils.lazy_loader import LazyLoader
from external.core.utils.text_split import split_text
from external.core.utils.token_generator import generate_token
from external.plugins.lip_sync.core.avatar import AvatarManager
from external.plugins.lip_sync.core.models import AvatarWave2LipModel
from external.plugins.text2speech import Text2Speech
from external.tools.video_stream import video_stream

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

avatar_manager = AvatarManager(AvatarWave2LipModel())
speech_loader = LazyLoader(Text2Speech, force_load=True)
video_buffers = {}
audio_buffers = {}
audio_video_map = {}


class AudioRequest:
    def __init__(self, text, voice_id):
        self.text_gen = split_text(text, 200)
        self.voice_id = voice_id

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.text_gen, None)


def register_video_buffer(audio_path, persona):
    token = generate_token()
    avatar = avatar_manager.get_avatar(persona)
    video_buffers[token] = avatar.video_buffer(audio_path)
    return token


def register_audio_buffer(text, voice_id, persona):
    token = generate_token()
    audio_request = AudioRequest(text, voice_id)
    audio_buffers[token] = audio_request
    text_request = next(audio_request)
    print("generating text for voice {}".format(text_request))
    video_token = request_audio_stream(text_request, audio_request.voice_id, persona)
    audio_video_map[token] = video_token
    return token


def request_audio_stream(text, voice_id, persona):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        speech_loader.get().convert(text, voice_id).as_file(temp_file.name)
        token = register_video_buffer(temp_file.name, persona)
    return token


def get_next_clip(audio_token, persona):
    if audio_token not in audio_buffers:
        return None
    video_token = audio_video_map[audio_token]
    next_buffer = next(video_buffers[video_token], None)
    if next_buffer is None:
        # this means video buffer of a text chunk ended, we have to generate a new one
        del video_buffers[video_token]
        # link the video token with the text token, removing the existing one and keeping only the new one
        del audio_video_map[audio_token]
        audio_request: AudioRequest = audio_buffers[audio_token]
        audio_request_text = next(audio_request, None)
        if audio_request_text is None:
            # this means that the full video is generated
            del audio_buffers[audio_token]
            return None
        print("generating audio for request:", audio_request_text)
        new_video_token = request_audio_stream(audio_request_text, audio_request.voice_id, persona)
        audio_video_map[audio_token] = new_video_token
        next_buffer = next(video_buffers[new_video_token], None)
    return next_buffer


def stream_clip(clip: ImageSequenceClip):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.close()
        clip.write_videofile(tmpfile.name, codec="libx264")
        yield open(tmpfile.name, mode="rb").read()
        os.remove(tmpfile.name)


@app.get("/get_tokens")
async def get_tokens() -> dict:
    return {'in_progress_video': list(video_buffers.keys()), 'request_map': audio_video_map}


@app.get("/request")
async def request_avatar(text: str, voice_id: int, persona: str, as_token=True) -> dict:
    token = register_audio_buffer(text, voice_id, persona)
    if as_token:
        others = await get_tokens()
        return {'token': token, **others}
    return stream_next(token)


@app.get("/idle")
def idle(persona):
    avatar = avatar_manager.get_avatar(persona)
    return video_stream(avatar.get_idle_stream())


@app.get("/stream_next")
async def stream_next(token):
    try:
        clip = get_next_clip(token, 'lisa_casual_720_pl')
        if clip is None:
            return "no more clips"
        return StreamingResponse(stream_clip(clip), media_type="video/mp4")
    except Exception as e:
        raise e


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
