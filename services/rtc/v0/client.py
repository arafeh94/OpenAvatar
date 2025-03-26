import asyncio
import json

from aiortc import RTCPeerConnection, RTCSessionDescription
import requests


async def run():
    response = requests.get("http://localhost:8000/register").json()
    sdp = RTCSessionDescription(response['sdp']['sdp'], response['sdp']['type'])
    token = response['token']

    peer = RTCPeerConnection()
    await peer.setRemoteDescription(sdp)
    await peer.setLocalDescription(await peer.createAnswer())
    sdp = json.dumps({'sdp': peer.localDescription.sdp, 'type': peer.localDescription.type})
    confirmation = requests.get(f"http://localhost:8000/confirm?token={token}&sdp={sdp}").json()
    print(confirmation)

    @peer.on("track")
    def on_track(event):
        print("Track detected")

    await asyncio.Event().wait()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
