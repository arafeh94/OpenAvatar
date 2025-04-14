import asyncio

from core.plugins.lip_sync.core.avatar_extentions import AvatarManager, avatar_file_writer
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech

from services.rtc.src.tracks.avatar_player import AvatarMediaPlayer, MediaSink

tts = MicrosoftText2Speech()
avatar_manager = AvatarManager(AvatarWave2LipModel(), tts)
# text = "Healthcare in the United States is a complex and multifaceted system that combines public and private elements, presenting both opportunities and challenges for individuals, policymakers, and the economy. It is often described as one of the most advanced healthcare systems in the world in terms of medical technology, research, and access to specialized care. However, it is also marked by disparities in access, affordability, and outcomes, which contribute to ongoing debates about reform and improvement. The healthcare system is largely dominated by private insurance companies, with a significant portion of the population receiving coverage through employer-sponsored insurance plans. In addition to these private plans, there are public health programs like Medicaid and Medicare that provide coverage for low-income individuals and those over the age of 65, respectively. However, a substantial number of Americans remain uninsured or underinsured, which has led to concerns about the overall effectiveness and equity of the system."
text = "Hello Samira, how are you?"

print("model loaded")
buffer = avatar_manager.tts_buffer("lisa_casual_720_pl", text, voice_id=7406)
print("buffer generated")


async def main():
    media_player = AvatarMediaPlayer()
    media_player.start(buffer)
    await MediaSink(media_player.video, media_player.audio).start()
    await asyncio.sleep(10)
    print("finished")


loop = asyncio.new_event_loop()
try:
    loop.run_until_complete(main())
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    loop.close()
