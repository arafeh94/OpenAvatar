import pickle
import numpy as np
from pydub import AudioSegment

# Load frames
frames = pickle.load(open("frames.pkl", "rb"))

all_audio = AudioSegment.silent(duration=0)  # empty AudioSegment

for saved_frame in frames:
    samples = saved_frame['samples'][0]  # mono
    # Convert int16 numpy array to bytes
    audio_bytes = samples.tobytes()

    # Create AudioSegment
    seg = AudioSegment(
        data=audio_bytes,
        sample_width=2,  # 16-bit = 2 bytes
        frame_rate=saved_frame['sample_rate'],
        channels=1
    )
    all_audio += seg

# Export to MP3
all_audio.export("output.mp3", format="mp3")
