# uvicorn services.test2speech.app:app
import asyncio
import io
import shutil
import soundfile as sf
import numpy as np
from fastapi.responses import StreamingResponse
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.io.wavfile import write
from transformers import pipeline
from datasets import load_dataset

from external.core.utils.lazy_loader import LazyLoader
from external.plugins.text2speech import Text2Speech

app = FastAPI()
speech_loader = LazyLoader(Text2Speech, force_load=True)


class SpeechRequest(BaseModel):
    text: str


# http://127.0.0.1:8000/generate_speech/Hello%20World/7306
@app.get("/generate_speech/{text}/{voice_id}", operation_id="generate_speech", tags=['speech'],
         response_class=StreamingResponse)
async def generate_speech(text: str, voice_id: int = 7306) -> StreamingResponse:
    return speech_loader.get().convert(text, voice_id).as_streaming_response()


# curl -X POST "http://127.0.0.1:8000/generate_speech_post/7306"
# -H "Content-Type: application/json" -d '{"text": "Hello world"}'
# --output "output_audio.wav"
@app.post("/generate_speech_post/{voice_id}", operation_id="generate_speech_post", tags=['speech'],
          response_class=StreamingResponse)
async def generate_speech_post(voice_id: int, request: SpeechRequest):
    text = request.text
    return await generate_speech(text, voice_id)
