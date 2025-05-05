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
