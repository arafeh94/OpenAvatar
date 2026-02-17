import asyncio
from pympler import asizeof
from core.plugins.lip_sync.core.avatar_extentions import AvatarManager

avatar_manager = AvatarManager()

print("model loaded")
while True:
    prompt = "hellooo, how you are doing?"
    if prompt == "p":
        break
    buffer = avatar_manager.tts_buffer("lisa_casual_720_pl", prompt)
    for frames, audio, text in buffer:
        for frame in frames:
            print(frame)
        print(audio)
        print(text)
    buffer.stop()
input("Press Enter to exit...")
