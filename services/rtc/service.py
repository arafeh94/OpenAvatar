import json
import uvicorn
from aiortc import RTCSessionDescription
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from core.tools.token_generator import generate_token
from core.tools import utils
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
    del AppContext().peers[token]
    print("peer {} closed".format(token))


@app.get("/register")
async def register():
    token = generate_token()
    server = ServerPeer(token, on_peer_close)
    sdp = await server.offer()
    AppContext().peers[token] = server
    return {"sdp": sdp, 'token': token}


@app.get("/confirm")
async def confirm(token, sdp):
    if token in AppContext().peers.keys():
        server = AppContext().peers[token]
        client_sdp = json.loads(sdp)
        client_sdp = RTCSessionDescription(client_sdp['sdp'], client_sdp['type'])
        await server.accept(client_sdp)
        return {"status": "accepted"}
    return {"status": "not registered"}


@app.get("/broadcast")
async def broadcast(message):
    for peer in AppContext().peers.values():
        peer.send_message(message)
    return {"status": "accepted"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
