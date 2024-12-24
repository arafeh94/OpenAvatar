from moviepy import ImageSequenceClip

from external.plugins.lip_sync.core.avatar import Avatar
from external.plugins.lip_sync.core.models import AvatarWave2LipModel

model = AvatarWave2LipModel()
avatar = Avatar("lisa_casual_720_pl", model)
avatar.init()

audio_path = "../files/harvard.wav"

bfs = avatar.video_buffer(audio_path)
clip: ImageSequenceClip = next(bfs)
clip.preview()


# buffer = BytesIO()
# clip.write_videofile(buffer, format="mp4", codec="libx264", audio=False)
# buffer.seek(0)
