import asyncio

from aiortc import RTCPeerConnection, MediaStreamTrack


def create_channel(peer):
    channel = peer.createDataChannel('chat')

    @channel.on("open")
    def channel_open():
        print("Open")
        channel.send("samira")

    return channel


p1 = RTCPeerConnection()
p1.createDataChannel('chat')
p1.setLocalDescription(p1.createOffer())

p2 = RTCPeerConnection()
p2.setRemoteDescription(p1.localDescription)
p2.setLocalDescription(p2.createOffer())

p1.setRemoteDescription(p2.localDescription)


class RTCPeerBuilder:
    def __init__(self):
        self.peer = None

    class Builder:
        def __init__(self, peer: RTCPeerConnection):
            self.peer = peer

        def data_channel(self, channel_name):
            self.peer.createDataChannel(channel_name)
            return self

        def track(self, track: MediaStreamTrack):
            self.peer.addTrack(track)

        async def build(self):
            await self.peer.setLocalDescription(await p1.createOffer())

            return RTCPeerConnection(self.peer)


async def run():
    p1 = RTCPeerConnection()
    create_channel(p1)
    await p1.setLocalDescription(await p1.createOffer())
    p2 = RTCPeerConnection()
    await p2.setRemoteDescription(p1.localDescription)
    await p2.setLocalDescription(await p2.createAnswer())

    await p1.setRemoteDescription(p2.localDescription)

    @p2.on("datachannel")
    def datachannel(channel):
        @channel.on("message")
        def message(msg):
            print(msg)

    await asyncio.sleep(2)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        pass
