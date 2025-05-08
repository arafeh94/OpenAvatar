from core.plugins.text2speech import ElevenLabsText2Speech

tts = ElevenLabsText2Speech()
a = tts.convert("hello how are you?")
audio = a.as_file('test.wav', format='wav')
