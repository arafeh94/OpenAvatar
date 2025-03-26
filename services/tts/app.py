# uvicorn services.test2speech.app:app
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from pydantic import BaseModel

from core.utils import LazyLoader
from core.plugins.text2speech import Text2Speech

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
