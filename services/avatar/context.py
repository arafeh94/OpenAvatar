from future.moves import sys

from external.core.utils.lazy_loader import LazyLoader
from external.core.utils.text_split import split_text
from external.plugins.lip_sync.core.avatar import AvatarManager
from external.plugins.lip_sync.core.models import AvatarWave2LipModel
from external.plugins.text2speech import Text2Speech


class AvatarServiceDataManager:
    def __init__(self):
        self.avatar_manager = AvatarManager(AvatarWave2LipModel())
        self.speech_loader = LazyLoader(Text2Speech, force_load=True)
        self.video_buffers = {}
        self.audio_buffers = {}
        self.audio_video_map = {}
        self.preloaded_video = {}
        self.preloading_threads = {}
        self.token_persona = {}

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
