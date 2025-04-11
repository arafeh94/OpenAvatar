import numpy as np

from core.interfaces.va import Audio
from core.plugins.lip_sync.core.avatar import AvatarManager
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech
import cv2
import time
from core.tools.file_tools import VideoWriter

text = 'The healthcare the United States is a complex and multifaceted system that combines public and private elements, presenting both opportunities and challenges for individuals, policymakers, and the economy. It is often described as one of the most advanced healthcare systems in the world in terms of medical technology, research, and access to specialized care. However, it is also marked by disparities in access, affordability, and outcomes, which contribute to ongoing debates about reform and improvement. The united healthcare system is largely dominated by private insurance companies, with a significant portion of the population receiving coverage through employer-sponsored insurance plans. In addition to these private plans, there are public health programs like Medicaid and Medicare that provide coverage for low-income individuals and those over the age of 65, respectively. However, a substantial number of Americans remain uninsured or underinsured, which has led to concerns about the overall effectiveness and equity of the system.'
# text = "Hello how are you"
avatar_manager = AvatarManager(AvatarWave2LipModel(), MicrosoftText2Speech())
buffer = avatar_manager.tts_buffer('lisa_casual_720_pl', text, voice_id=7406)
writer = VideoWriter('out.avi', 24, (640, 480))
first_audio = None
t = time.time()
for video, audio in buffer:
    print(time.time() - t)
    t = time.time()
    audio: Audio
    print("{}s".format(len(audio.samples) / audio.sampling_rate))
    # if first_audio is None:
    #     first_audio = audio
    # first_audio.extends(audio)
    # for frame in video:
    #     writer.write(frame)
# writer.release()
# first_audio.write('out.wav')
