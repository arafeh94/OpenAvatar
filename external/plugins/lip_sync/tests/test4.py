import io
import os
import threading

from avatar.avatar_only.atomic_id import AtomicID
from avatar.avatar_only.core import Avatar, AvatarWave2LipModel
import matplotlib.pyplot as plt

import cv2
import ffmpeg
import numpy as np
import subprocess
from scipy.io.wavfile import write as write_wav
from moviepy import *

from avatar.avatar_only.manifest import Manifest

model = AvatarWave2LipModel()

audio_path = "../files/harvard.wav"

manifest = Manifest()


def video_writer_thread(audio_path):
    avatar = Avatar("lisa_casual_720_pl", model)  # Assuming model is defined elsewhere
    avatar.init()
    avatar.video_writer(audio_path, f'_out1{AtomicID().fetch()}.mp4')


def run_video_writer_async(audio_path):
    video_writer_thread_instance = threading.Thread(target=video_writer_thread, args=(audio_path,))
    video_writer_thread_instance.start()


audio_path = "../files/harvard.wav"
print("running video_writer_thread")
run_video_writer_async(audio_path)
run_video_writer_async(audio_path)
run_video_writer_async(audio_path)
run_video_writer_async(audio_path)
