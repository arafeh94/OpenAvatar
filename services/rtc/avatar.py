from aiortc import RTCPeerConnection


async def peer():
    pc = RTCPeerConnection()
    channel = pc.createDataChannel("chat")

    @pc.on("datachannel")
    def on_datachannel(data_channel):
        @data_channel.on("message")
        def on_message(message):
            print(message)

    @channel.on("open")
    def on_open():
        print("Connected")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

