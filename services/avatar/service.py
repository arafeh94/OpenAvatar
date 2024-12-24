import asyncio
import os
import tempfile
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from moviepy import ImageSequenceClip

from external.core.utils.lazy_loader import LazyLoader
from external.core.utils.text_split import split_text
from external.core.utils.token_generator import generate_token
from external.plugins.lip_sync.core.avatar import Avatar
from external.plugins.lip_sync.core.models import AvatarWave2LipModel
from external.plugins.text2speech import Text2Speech

app = FastAPI()
avatar_model = AvatarWave2LipModel()
speech_loader = LazyLoader(Text2Speech, force_load=True)
video_buffers = {}
audio_buffers = {}
audio_video_map = {}


class AudioRequest:
    def __init__(self, text, voice_id):
        self.text_gen = split_text(text, 200)
        self.voice_id = voice_id

    def __next__(self):
        return next(self.text_gen, None)


def register_video_buffer(audio_path, avatar_cache="lisa_casual_720_pl"):
    token = generate_token()
    avatar = Avatar(avatar_cache, avatar_model)
    avatar.init()
    video_buffers[token] = avatar.video_buffer(audio_path)
    return token


def register_audio_buffer(text, voice_id):
    token = generate_token()
    audio_request = AudioRequest(text, voice_id)
    audio_buffers[token] = audio_request
    text_request = next(audio_request)
    print("generating text for voice {}".format(text_request))
    video_token = request_audio_stream(text_request, audio_request.voice_id)
    audio_video_map[token] = video_token
    return token


def request_audio_stream(text, voice_id):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        speech_loader.get().convert(text, voice_id).as_file(temp_file.name)
        token = register_video_buffer(temp_file.name)
    return token


def get_next_clip(audio_token):
    if audio_token not in audio_buffers:
        raise KeyError(f"Requested token [{audio_token}] does not exist. Should register first")
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
        new_video_token = request_audio_stream(audio_request_text, audio_request.voice_id)
        audio_video_map[audio_token] = new_video_token
        next_buffer = next(video_buffers[new_video_token], None)
    return next_buffer


def stream_clip(clip: ImageSequenceClip):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.close()
        clip.write_videofile(tmpfile.name, codec="libx264")
        yield open(tmpfile.name, mode="rb").read()
        os.remove(tmpfile.name)


@app.get("/request")
async def request_avatar_old(text: str, voice_id: int) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        speech_loader.get().convert(text, voice_id).as_file(temp_file.name)
        token = register_video_buffer(temp_file.name)
    return {'token': token, 'in_progress': list(video_buffers.keys())}


@app.get("/request")
async def request_avatar(text: str, voice_id: int) -> dict:
    token = register_audio_buffer(text, voice_id)
    return {'token': token, 'in_progress_video': list(video_buffers.keys()),
            'in_progress_audio': list(audio_buffers.values())}


@app.get("/stream_next")
async def run_main(token):
    try:
        clip = get_next_clip(token)
        while clip is not None:
            raise HTTPException(status_code=404, detail="Text is fully exhausted")
        return StreamingResponse(stream_clip(clip), media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=404, detail="Requested token does not exists")


if __name__ == "__main__":
    # uvicorn.run(app, host="localhost", port=8000)
    #     text = """Technology has significantly impacted various aspects of our lives, from how we communicate to how we work and learn. It has led to the development of new industries, improved productivity, and increased accessibility to information. However, with these advancements come new challenges, such as the growing digital divide. While some regions and populations have easy access to technology, others are left behind, unable to fully participate in the digital age.
    # In education, technology has transformed traditional learning methods. Online courses, virtual classrooms, and digital resources have made education more flexible and accessible. Students can now learn from anywhere in the world, provided they have the necessary technology. Yet, this shift to digital learning has its downsides, including concerns over the lack of face-to-face interaction and the potential for widening educational inequalities.
    # As technology continues to advance, it is essential that we address these challenges. Ensuring equal access to technology and preserving the value of human interaction in education will be crucial to shaping a more inclusive and balanced future.
    #     """
    text = "hello samira"
    tokens_dict = asyncio.run(request_avatar(text, 1))
    token = tokens_dict['token']
    while True:
        clip = get_next_clip(token)
        if clip is None:
            break
        clip.preview()
