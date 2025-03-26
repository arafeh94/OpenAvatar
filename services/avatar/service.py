import logging
import os
import tempfile
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from moviepy import ImageSequenceClip
from starlette.middleware.cors import CORSMiddleware

from core.tools.token_generator import generate_token
from core.plugins.lip_sync.core.avatar_mp4 import video_buffer
from core.tools import utils
from core.tools.memory_profiler import getsize
from core.tools.video_stream import video_stream
from services.avatar.context import AvatarServiceDataManager, AudioRequest

utils.enable_logging()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger('Avatar-Service')
data_manager = AvatarServiceDataManager()


def register_video_buffer(audio_path, persona):
    token = generate_token()
    avatar = data_manager.avatar_manager.get_avatar(persona)
    data_manager.video_buffers[token] = video_buffer(avatar, audio_path)
    return token


def register_audio_buffer(text, voice_id, persona):
    token = generate_token()
    audio_request = AudioRequest(text, voice_id)
    data_manager.audio_buffers[token] = audio_request
    text_request = next(audio_request)
    logger.info("Text-Voice request: {}".format(text_request))
    video_token = request_audio_stream(text_request, audio_request.voice_id, persona)
    data_manager.audio_video_map[token] = video_token
    return token


def request_audio_stream(text, voice_id, persona):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        data_manager.speech_loader.get().convert(text, voice_id).as_file(temp_file.name)
        token = register_video_buffer(temp_file.name, persona)
    return token


def non_blocking_preload(audio_token, persona):
    def preload():
        data_manager.preloaded_video[audio_token] = get_next_clip(audio_token, persona)

    thread = threading.Thread(target=preload)
    data_manager.preloading_threads[audio_token] = thread
    thread.start()


def get_preloaded_video(token):
    if token not in data_manager.preloaded_video:
        return False
    if token in data_manager.preloading_threads:
        thread = data_manager.preloading_threads[token]
        thread.join()
    data_manager.preloading_threads.pop(token)
    return data_manager.preloaded_video.pop(token)


def get_next_clip(audio_token, persona, preload=False):
    if audio_token not in data_manager.audio_buffers:
        return None
    next_buffer = get_preloaded_video(audio_token)
    if not next_buffer:
        video_token = data_manager.audio_video_map[audio_token]
        next_buffer = next(data_manager.video_buffers[video_token], None)
        if next_buffer is None:
            # this means video buffer of a text chunk ended, we have to generate a new one
            del data_manager.video_buffers[video_token]
            # link the video token with the text token, removing the existing one and keeping only the new one
            del data_manager.audio_video_map[audio_token]
            audio_request: AudioRequest = data_manager.audio_buffers[audio_token]
            audio_request_text = next(audio_request, None)
            if audio_request_text is None:
                # this means that the full video is generated
                del data_manager.audio_buffers[audio_token]
                return None
            logger.info("Generating TTS for text: {}".format(audio_request_text))
            new_video_token = request_audio_stream(audio_request_text, audio_request.voice_id, persona)
            data_manager.audio_video_map[audio_token] = new_video_token
            next_buffer = next(data_manager.video_buffers[new_video_token], None)
    if preload:
        # start a preloading process in background to avoid the video from cutting
        non_blocking_preload(audio_token, persona)
    return next_buffer


def stream_clip(clip: ImageSequenceClip):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.close()
        clip.write_videofile(tmpfile.name, codec="libx264")
        yield open(tmpfile.name, mode="rb").read()
        os.remove(tmpfile.name)


@app.get("/get_tokens")
async def get_tokens() -> dict:
    return {'in_progress_video': list(data_manager.video_buffers.keys()), 'request_map': data_manager.audio_video_map}


@app.get("/request")
async def request_avatar(text: str, voice_id: int, persona: str, as_token=True) -> dict:
    token = register_audio_buffer(text, voice_id, persona)
    if as_token:
        others = await get_tokens()
        return {'token': token, **others}
    return stream_next(token)


@app.get("/idle")
def idle(persona):
    avatar = data_manager.avatar_manager.get_avatar(persona)
    return video_stream(avatar.get_idle_stream())


@app.get("/stream_next")
async def stream_next(token, persona):
    try:
        logger.info("Used Memory: {}".format(getsize(data_manager)))
        clip = get_next_clip(token, persona, preload=True)
        if clip is None:
            return "no more clips"
        return StreamingResponse(stream_clip(clip), media_type="video/mp4")
    except Exception as e:
        raise e


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
