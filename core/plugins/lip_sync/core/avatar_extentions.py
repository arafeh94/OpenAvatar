import logging
import os
import platform
import subprocess
from typing import Generator, Union, Any
import cv2
from core.interfaces.base_tts import Text2Speech
from core.interfaces.va import Audio
from core.plugins.lip_sync.core.avatar import Avatar
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import ElevenLabsText2Speech
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


def avatar_file_writer(output_path, avatar_buffer, audio, width_height=(1280, 720), fps=24):
    ts = time.strftime("%Y%m%d_%H%M%S")
    rid = random.randint(0, 99999)

    temp_video = f"_temp_avatar_{ts}_{rid}.mp4"
    temp_audio = f"_temp_audio_{ts}_{rid}.wav"

    out = cv2.VideoWriter(
        temp_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        width_height
    )
    if not out.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open. Check codec support and output path.")

    for frame in avatar_buffer:
        if frame is None:
            continue
        if frame.shape[1] != width_height[0] or frame.shape[0] != width_height[1]:
            frame = cv2.resize(frame, width_height, interpolation=cv2.INTER_LINEAR)
        out.write(frame)

    out.release()

    audio.write(temp_audio)  # this should actually write a WAV/PCM file

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", temp_audio,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)

    os.remove(temp_video)
    os.remove(temp_audio)
