import fractions
import logging
import os
import platform
import subprocess
from typing import Generator, Union, Any, Tuple, List, Callable
import cv2
import numpy as np
from aiortc.mediastreams import AUDIO_PTIME
from av import AudioFrame, VideoFrame
from av.frame import Frame

from core.interfaces.base_tts import Text2Speech
from core.interfaces.va import Audio
from core.plugins.lip_sync.core.avatar import Avatar
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech, ElevenLabsText2Speech
from core.tools import utils
from core.tools.async_generator import NonBlockingLookaheadGenerator
from core.tools.text_split import TextSampler

from manifest import Manifest
import time
import random


class AvatarTTS:
    def __init__(self, avatar: Avatar, tts_convertor: Text2Speech):
        self.tts_convertor = tts_convertor
        self.avatar = avatar
        self.logger = logging.getLogger(self.__class__.__name__)

    def tts(self, text, **kwargs) -> Generator[Any, Audio, Union[str, None]]:
        max_batch_size = kwargs.get('max_text_sample', Manifest().query('tts.max_text_sample', 200))
        text_sampler = TextSampler(text, max_batch_size)
        while True:
            speech_text: str = next(text_sampler)
            if speech_text is None:
                break
            audio = self.tts_convertor.convert(speech_text, **kwargs).as_audio()
            yield self.avatar.stream(audio, **kwargs), audio, speech_text

    # noinspection PyTypeChecker
    def buffer(self, text, **kwargs) -> Generator[Any, Audio, Union[str, None]]:
        return NonBlockingLookaheadGenerator(self.tts(text, **kwargs), 'tts_buffer')


class AvatarManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AvatarManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self.avatar_cache = {}
        self.avatar_model = AvatarWave2LipModel()
        self.tts_convertor = ElevenLabsText2Speech()
        self._initialized = True

    def get_avatar(self, persona):
        if persona not in self.avatar_cache:
            avatar = Avatar(persona, self.avatar_model)
            avatar.init()
            self.avatar_cache[persona] = avatar
        else:
            avatar = self.avatar_cache[persona]
        return avatar

    def tts_buffer(self, persona, text, **kwargs) -> Generator[Any, Audio, Union[str, None]]:
        """
        Return a generator where each yield return the frames and the audio assigned to these frames.
        Be careful, this method takes too much time. Better run it inside a thread
        Args:
            persona: avatar files
            text: the text to repeat (convert to speach and generate lip-synced frames)
            **kwargs: additional arguments passed to tts
                voice_id=7406

        Returns: yield frames, audio, text

        """
        avatar = self.get_avatar(persona)
        tts_buffer = AvatarTTS(avatar, self.tts_convertor)
        return tts_buffer.buffer(text, **kwargs)


def avatar_file_writer(output_path, avatar_buffer: Avatar.AvatarBuffer, audio: Audio, width_height=(1280, 720)):
    temp_avatar = f'_temp_avatar_{time.strftime("%Y%m%d_%H%M%S")}_{random.randint(0, 99999)}.avi'
    temp_audio = f'_temp_audio_{time.strftime("%Y%m%d_%H%M%S")}_{random.randint(0, 99999)}.wav'
    print("writing frames started")
    # noinspection PyUnresolvedReferences
    out = cv2.VideoWriter(temp_avatar, cv2.VideoWriter_fourcc(*'DIVX'), 24, width_height)
    for frame in avatar_buffer:
        out.write(frame)
    out.release()
    audio.write(temp_audio)
    command = f'ffmpeg -y -i {temp_audio} -i {temp_avatar} -strict -2 -q:v 1 {output_path}'
    subprocess.call(command, shell=platform.system() != 'Windows')
    os.remove(temp_avatar)
    os.remove(temp_audio)


class AvatarVideoDecoder:
    CLOCK_RATE = 90_000
    FPS = 24

    @staticmethod
    def decode(stream: Tuple[Any, Audio], frame_decoded: Callable[[Frame], None]):
        AVD = AvatarVideoDecoder
        video_buffer, audio_frames = stream
        video_buffer, audio_frames = AVD.TSVideo(video_buffer), AVD.TSAudio(audio_frames)
        audio_time, video_time = 0, 0
        v_index, a_index = 0, 0
        while True:
            try:
                if video_time <= audio_time:
                    video_frame = next(video_buffer)
                    video_time = float(video_frame.pts * video_frame.time_base)
                    v_index += 1
                    frame_decoded(video_frame)
                else:
                    audio_frame = next(audio_frames)
                    audio_time = float(audio_frame.pts * audio_frame.time_base)
                    a_index += 1
                    frame_decoded(audio_frame)
            except StopIteration:
                break

    class TSVideo:
        def __init__(self, frame_buffer):
            self.frame_buffer = frame_buffer
            self.pts = 0
            self.pts_increment = int(AvatarVideoDecoder.CLOCK_RATE / AvatarVideoDecoder.FPS)

        def __iter__(self):
            return self

        def __next__(self):
            frame = next(self.frame_buffer)
            if frame is None:
                raise StopIteration
            av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            av_frame.time_base = fractions.Fraction(1, AvatarVideoDecoder.CLOCK_RATE)
            av_frame.pts = self.pts
            self.pts += self.pts_increment
            return av_frame

    class TSAudio:
        def __init__(self, audio: Audio):
            self.audio = audio
            self.pts = 0
            self.audio_time_base = fractions.Fraction(1, audio.sampling_rate)
            self.batches = iter(utils.create_batches(audio.samples, int(audio.sampling_rate * AUDIO_PTIME)))

        def __iter__(self):
            return self

        def __next__(self):
            audio_samples = next(self.batches)
            block = (np.array(audio_samples) / max(1, np.max(np.abs(audio_samples))) * 32767).astype(np.int16)
            av_frame = AudioFrame.from_ndarray(block.reshape(1, -1), format='s16', layout='mono')
            av_frame.sample_rate = self.audio.sampling_rate
            av_frame.pts = self.pts
            av_frame.time_base = self.audio_time_base
            self.pts += len(audio_samples)
            return av_frame

    @staticmethod
    def idle(persona, frame_index, pts=0) -> List[Frame]:
        frame = AvatarManager().get_avatar(persona).get_frame(frame_index)
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.time_base = fractions.Fraction(1, AvatarVideoDecoder.CLOCK_RATE)
        av_frame.pts = pts
        return [av_frame]

    @staticmethod
    def silence(sample_rate: int, duration_sec: float, pts: int = 0) -> List[Frame]:
        total_samples = int(sample_rate * duration_sec)
        max_samples_per_frame = int(sample_rate * AUDIO_PTIME)
        audio_time_base = fractions.Fraction(1, sample_rate)

        frames = []
        remaining = total_samples

        while remaining > 0:
            current_samples = min(remaining, max_samples_per_frame)
            block = np.zeros((1, current_samples), dtype=np.int16)

            av_frame = AudioFrame.from_ndarray(block, format='s16', layout='mono')
            av_frame.sample_rate = sample_rate
            av_frame.pts = pts
            av_frame.time_base = audio_time_base

            frames.append(av_frame)

            pts += current_samples
            remaining -= current_samples

        return frames
