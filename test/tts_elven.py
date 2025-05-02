import io
import os
import uuid

import numpy as np
from av import AudioResampler
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

from core.interfaces.va import Audio
from core.plugins.text2speech import ElevenLabsText2Speech

tts = ElevenLabsText2Speech()
a = tts.convert("hello how are you?")
audio = a.as_file('test.wav', format='wav')
