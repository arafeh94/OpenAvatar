from core.plugins.lip_sync.core.avatar_extentions import AvatarManager
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech

tts = MicrosoftText2Speech()
a = tts.convert("hello samira", voice_id=7406).as_audio()
print(a)