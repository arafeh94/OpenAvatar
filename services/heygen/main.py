import asyncio
import json
import logging
import typing

import uvicorn
from aiortc import RTCSessionDescription
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from core.tools import utils
from core.tools.token_generator import generate_token
from services.heygen.heygen import Heygen
from services.heygen.peer import HeygenPeer
from services.rtc.context import AppContext

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
        heygen = Heygen(persona)
        server = HeygenPeer(token, heygen, on_peer_close)
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
            server: HeygenPeer = AppContext().peers[token]
            client_sdp = json.loads(sdp)
            client_sdp = RTCSessionDescription(client_sdp['sdp'], client_sdp['type'])
            logging.info("Accepting Client {}, with SDP: {}".format(token, client_sdp))
            await server.accept(client_sdp)
            return {"status": "200", "message": "connection accepted"}
        except Exception as e:
            return {"status": "500", "message": str(e)}
    return {"status": "404", "message": "peer not found, you should register first"}


@app.get("/broadcast")
async def broadcast(message):
    for peer in AppContext().peers.values():
        peer.send_message(message)
    return {"status": "200", "message": "message broadcasted"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
