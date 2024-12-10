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

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
speech_model = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device)
speech_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")


class SpeechRequest(BaseModel):
    text: str


# http://127.0.0.1:8000/generate_speech/Hello%20World/7306
@app.get("/generate_speech/{text}/{voice_id}", operation_id="generate_speech", tags=['speech'],
         response_class=StreamingResponse)
async def generate_speech(text: str, voice_id: int = 7306) -> StreamingResponse:
    voice_id = int(voice_id) if voice_id is not None else 7306
    speaker_embeddings = torch.tensor(speech_dataset[voice_id]["xvector"]).unsqueeze(0)
    speech = speech_model(text, forward_params={"speaker_embeddings": speaker_embeddings})

    buffer = io.BytesIO()
    sf.write(buffer, speech["audio"], samplerate=speech["sampling_rate"], format="WAV")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")


# curl -X POST "http://127.0.0.1:8000/generate_speech_post/7306"
# -H "Content-Type: application/json" -d '{"text": "Hello world"}'
# --output "output_audio.wav"
@app.post("/generate_speech_post/{voice_id}", operation_id="generate_speech_post", tags=['speech'],
          response_class=StreamingResponse)
async def generate_speech_post(voice_id: int, request: SpeechRequest):
    text = request.text
    return await generate_speech(text, voice_id)
