import time

from external.plugins.lip_sync.core.avatar import Avatar
from external.plugins.lip_sync.core.models import AvatarWave2LipModel

model = AvatarWave2LipModel()
avatar = Avatar("lisa_casual_720_pl", model)
avatar.init()

audio_path = "../files/harvard.wav"

t = time.time()
buffer = avatar.video_buffer(audio_path)
for i in buffer:
    print(time.time() - t)
    t = time.time()
    i.preview()

# clips = [v for v in avatar.video_buffer(audio_path)]
# print([c.duration for c in clips], '\n', 'sum:' + str(sum([c.duration for c in clips])))
# final = concatenate_videoclips(clips)
# final.preview()
