import json
import uvicorn
from aiortc import RTCSessionDescription
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from core.utils.token_generator import generate_token
from core.plugins.lip_sync.core.avatar import AvatarManager
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech
from core.tools import utils
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

peers: dict[str, 'ServerPeer'] = {}
avatar_manager = AvatarManager(AvatarWave2LipModel(), MicrosoftText2Speech())


def on_peer_close(token):
    global peers
    del peers[token]


@app.get("/register")
async def register():
    token = generate_token()
    server = ServerPeer(token, on_peer_close, avatar_manager)
    sdp = await server.offer()
    peers[token] = server
    return {"sdp": sdp, 'token': token}


@app.get("/confirm")
async def confirm(token, sdp):
    if token in peers.keys():
        server = peers[token]
        client_sdp = json.loads(sdp)
        client_sdp = RTCSessionDescription(client_sdp['sdp'], client_sdp['type'])
        await server.accept(client_sdp)
        return {"status": "accepted"}
    return {"status": "not registered"}


@app.get("/broadcast")
async def broadcast(message):
    for peer in peers.values():
        peer.send_message(message)
    return {"status": "accepted"}


@app.get("/answer")
async def answer(message, token):
    if token not in peers.keys():
        return {"status": "not registered"}
    peer = peers[token]
    peer.stream(message, 'lisa_casual_720_pl')
    return {"status": "accepted"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
