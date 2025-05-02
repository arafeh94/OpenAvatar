from core.interfaces.va import Audio
from core.plugins.lip_sync.core.avatar_extentions import AvatarManager, avatar_file_writer

avatar_manager = AvatarManager()

avatar = avatar_manager.get_avatar('lisa_casual_720_pl')
audio = Audio.load('test.wav')
buffer = avatar.stream(audio)
avatar_file_writer("avatar2.mp4", buffer, audio)
