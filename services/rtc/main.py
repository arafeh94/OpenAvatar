import asyncio
import json
import logging
import shutil
import typing
from pathlib import Path

import uvicorn
from aiortc import RTCSessionDescription
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware

from core.plugins.face_detectors import YoloFaceDetector
from core.plugins.lip_sync.core.avatar_creator import AvatarCreator
from core.tools import utils
from core.tools.token_generator import generate_token
from services.rtc.context import AppContext
from services.rtc.src.peer import ServerPeer

utils.enable_logging()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def on_peer_close(token):
    AppContext().del_peer(token)
    print("peer {} closed".format(token))


@app.get("/register")
async def register(persona: str):
    try:
        token = generate_token()
        server = ServerPeer(token, persona, on_peer_close)
        logging.info("Offering RTC to client {}".format(token))
        sdp = await server.offer()
        AppContext().add_peer(token, server)
        return {"sdp": sdp, "token": token, "status": "202"}
    except Exception as e:
        return {"status": "500", "message": str(e)}


@app.get("/confirm")
async def confirm(token, sdp):
    if token in AppContext().peers.keys():
        try:
            server: ServerPeer = AppContext().peers[token]
            client_sdp = json.loads(sdp)
            client_sdp = RTCSessionDescription(client_sdp['sdp'], client_sdp['type'])
            logging.info("Accepting Client {}, with SDP: {}".format(token, client_sdp))
            await server.accept(client_sdp)
            server.player.start(None)
            return {"status": "200", "message": "connection accepted"}
        except Exception as e:
            return {"status": "500", "message": str(e)}
    return {"status": "404", "message": "peer not found, you should register first"}


@app.get("/broadcast")
async def broadcast(message):
    for peer in AppContext().peers.values():
        peer.send_message(message)
    return {"status": "200", "message": "message broadcasted"}


@app.post("/upload")
async def upload_file(persona: str, file: UploadFile = File(...)):
    BASE_DIR = Path("~/files")
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = BASE_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    creator = AvatarCreator(YoloFaceDetector())
    creator.create(persona, file_path)

    return {
        "filename": file.filename,
        "path": str(file_path)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
