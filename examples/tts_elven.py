from core.plugins.text2speech import ElevenLabsText2Speech

tts = ElevenLabsText2Speech()
a = tts.convert("hello how are you?", voice_id='21m00Tcm4TlvDq8ikWAM')
audio = a.as_file('test.wav', format='wav')
