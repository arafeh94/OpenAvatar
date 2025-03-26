from future.moves import sys

from core.utils import LazyLoader
from core.tools.text_split import split_text
from core.plugins.lip_sync.core.avatar import AvatarManager
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech


class AvatarServiceDataManager:
    def __init__(self):
        self.speech_loader = LazyLoader(MicrosoftText2Speech, force_load=True)
        self.avatar_manager = AvatarManager(AvatarWave2LipModel(), self.speech_loader.get())
        self.video_buffers = {}
        self.audio_buffers = {}
        self.audio_video_map = {}
        self.preloaded_video = {}
        self.preloading_threads = {}

    def memory(self):
        size_info = {}

        total_size_in_bytes = sys.getsizeof(self)
        total_size_in_mb = total_size_in_bytes / (1024 * 1024)
        size_info['total_size'] = round(total_size_in_mb, 2)

        for attribute_name, attribute_value in vars(self).items():
            attribute_size_in_bytes = sys.getsizeof(attribute_value)
            size_info[attribute_name] = round(attribute_size_in_bytes, 2)

        return size_info


class AudioRequest:
    def __init__(self, text, voice_id):
        self.text_gen = split_text(text, 200)
        self.voice_id = voice_id

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.text_gen, None)
