import sounddevice as sd
import librosa
import time

from core.plugins.lip_sync.core.avatar import Avatar, AvatarManager
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech, Audio

avatar_manager = AvatarManager(AvatarWave2LipModel(), MicrosoftText2Speech())


def run_input():
    while True:
        text = input('New Request?, Text: ')
        buffer = avatar_manager.tts_buffer('lisa_casual_720_pl', text, voice_id=7406)
        print("New request buffer generated!")
        while True:
            input('Next?')
            try:
                frames, audio = next(buffer)
                print(frames)
                print(audio)
            except StopIteration:
                print('No more frames')
                break


def run_latency(wait_play, write_files):
    while True:
        text = input('New Request?, Text: ')
        if text == '':
            break
        checkpoint = time.time()
        print("Processing...")
        buffer = avatar_manager.tts_buffer('lisa_casual_720_pl', text, voice_id=7406)
        checkpoint = time.time() - checkpoint
        print(f"Creating buffer took: {checkpoint}s")
        checkpoint = time.time()
        for i, (frame_buffer, audio) in enumerate(buffer):
            audio.write(f"./output{i}.wav")
            print(f"Audio and FrameGen took: {time.time() - checkpoint}s")
            checkpoint = time.time()
            next(frame_buffer)
            print(f"Main Frame {i} took: {time.time() - checkpoint}s")
            j = 0
            while True:
                try:
                    checkpoint = time.time()
                    next(frame_buffer)
                    print(f"Sub Frame ({j}) of Main({i}) took: {time.time() - checkpoint}s")
                    j += 1
                    if wait_play:
                        print("Waiting for video to finish playing...")
                        time.sleep(4)
                except StopIteration:
                    break

        print("--------")
        print("--------")


if __name__ == '__main__':
    run_latency(wait_play=False, write_files=False)
