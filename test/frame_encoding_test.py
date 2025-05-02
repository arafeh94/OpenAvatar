import fractions

import av
import io

import numpy as np
from aiortc.mediastreams import AUDIO_PTIME
from av import AudioFrame

from core.interfaces.va import Audio
from core.plugins.lip_sync.core.avatar_extentions import AvatarManager, avatar_file_writer
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech
import soundfile as sf


def create_batches(arr, batch_size):
    num_batches = len(arr) // batch_size + (1 if len(arr) % batch_size != 0 else 0)
    batches = [arr[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    return batches


tts = MicrosoftText2Speech()
avatar_manager = AvatarManager()
text = "Healthcare in the United States is a complex and multifaceted system that combines public and private elements, presenting both opportunities and challenges for individuals, policymakers, and the economy. It is often described as one of the most advanced healthcare systems in the world in terms of medical technology, research, and access to specialized care. However, it is also marked by disparities in access, affordability, and outcomes, which contribute to ongoing debates about reform and improvement. The healthcare system is largely dominated by private insurance companies, with a significant portion of the population receiving coverage through employer-sponsored insurance plans. In addition to these private plans, there are public health programs like Medicaid and Medicare that provide coverage for low-income individuals and those over the age of 65, respectively. However, a substantial number of Americans remain uninsured or underinsured, which has led to concerns about the overall effectiveness and equity of the system."

buffer = avatar_manager.tts_buffer("lisa_casual_720_pl", text, voice_id=7406)
video, audio, text = next(buffer)
video_frames = []
audio_frames = []

for i, frame in enumerate(video):
    av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
    av_frame.pts = int(90000 / 24 * (i + 1))
    av_frame.time_base = fractions.Fraction(1, 90000)
    video_frames.append(av_frame)

for audio_samples in create_batches(audio.samples, int(audio.sampling_rate * AUDIO_PTIME)):
    block = (np.array(audio_samples) / max(1, np.max(np.abs(audio_samples))) * 32767).astype(np.int16)
    av_frame = AudioFrame.from_ndarray(block.reshape(1, -1), format='s16', layout='mono')
    av_frame.sample_rate = audio.sampling_rate
    audio_frames.append(av_frame)
