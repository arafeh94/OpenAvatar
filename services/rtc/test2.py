import asyncio
import pickle
import time

import numpy as np
import soundfile as sf
from aiortc import MediaStreamTrack
from av import AudioFrame

audio_data, sample_rate = sf.read('../../files/harvard.wav', dtype='int16')
sf.write('harvard3.wav', audio_data, sample_rate)
