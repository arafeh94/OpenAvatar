import io
import os

from avatar.avatar_only.core import Avatar, AvatarWave2LipModel
import matplotlib.pyplot as plt

import cv2
import ffmpeg
import numpy as np
import subprocess
from scipy.io.wavfile import write as write_wav
from moviepy import *

model = AvatarWave2LipModel()
avatar = Avatar("lisa_casual_720_pl", model)
avatar.init()

audio_path = "../files/harvard.wav"

avatar.video_writer(audio_path, '_out1.mp4')
