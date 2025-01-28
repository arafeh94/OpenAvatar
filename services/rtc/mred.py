import asyncio

from aiortc import RTCPeerConnection


def create_channel(peer):
    channel = peer.createDataChannel('chat')

    @channel.on("open")
    def channel_open():
        print("Open")
        channel.send("samira")

    return channel


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
