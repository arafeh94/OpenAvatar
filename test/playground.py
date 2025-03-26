import sounddevice as sd
import librosa

from core.plugins.text2speech import MicrosoftText2Speech, Audio

audio_path = "../files/harvard.wav"

tts = MicrosoftText2Speech()
voice = tts.convert('hello samira', voice_id=7406).as_audio()
sd.play(voice.samples, samplerate=voice.sampling_rate, blocking=True)
print(voice)
# wav, sample_rate = librosa.core.load(audio_path, sr=16000)
# print(sample_rate)
# sd.play(wav, sample_rate)
# sd.wait()
