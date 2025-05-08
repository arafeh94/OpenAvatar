from core.interfaces.va import Audio
from core.plugins.lip_sync.core.avatar_extentions import AvatarManager, avatar_file_writer

audio_path = 'test.wav'
persona = 'lisa_casual_720_pl'
output = 'avatar.mp4'

if __name__ == '__main__':
    avatar_manager = AvatarManager()
    avatar = avatar_manager.get_avatar(persona)
    audio = Audio.load(audio_path)
    buffer = avatar.stream(audio)
    avatar_file_writer(output, buffer, audio)
