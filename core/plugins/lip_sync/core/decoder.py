import fractions
from typing import Any, Tuple, List, Callable

import numpy as np
from aiortc.mediastreams import AUDIO_PTIME
from av import AudioFrame, VideoFrame
from av.frame import Frame

from core.interfaces.va import Audio
from core.plugins.lip_sync.core.avatar_extentions import AvatarManager
from core.tools import utils


class AvatarVideoDecoder:
    CLOCK_RATE = 90_000
    FPS = 24

    @staticmethod
    def decode(stream: Tuple[Any, Audio], frame_decoded: Callable[[Frame], None]):
        AVD = AvatarVideoDecoder
        video_buffer, audio_frames = stream
        video_buffer, audio_frames = AVD.TSVideo(video_buffer), AVD.TSAudio(audio_frames)
        audio_time, video_time = 0, 0
        v_index, a_index = 0, 0
        while True:
            try:
                if video_time <= audio_time:
                    video_frame = next(video_buffer)
                    video_time = float(video_frame.pts * video_frame.time_base)
                    v_index += 1
                    frame_decoded(video_frame)
                else:
                    audio_frame = next(audio_frames)
                    audio_time = float(audio_frame.pts * audio_frame.time_base)
                    a_index += 1
                    frame_decoded(audio_frame)
            except StopIteration:
                break

    class TSVideo:
        def __init__(self, frame_buffer):
            self.frame_buffer = frame_buffer
            self.pts = 0
            self.pts_increment = int(AvatarVideoDecoder.CLOCK_RATE / AvatarVideoDecoder.FPS)

        def __iter__(self):
            return self

        def __next__(self):
            frame = next(self.frame_buffer)
            if frame is None:
                raise StopIteration
            av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            av_frame.time_base = fractions.Fraction(1, AvatarVideoDecoder.CLOCK_RATE)
            av_frame.pts = self.pts
            self.pts += self.pts_increment
            return av_frame

    class TSAudio:
        def __init__(self, audio: Audio):
            self.audio = audio
            self.pts = 0
            self.audio_time_base = fractions.Fraction(1, audio.sampling_rate)
            self.batches = iter(utils.create_batches(audio.samples, int(audio.sampling_rate * AUDIO_PTIME)))

        def __iter__(self):
            return self

        def __next__(self):
            audio_samples = next(self.batches)
            block = (np.array(audio_samples) / max(1, np.max(np.abs(audio_samples))) * 32767).astype(np.int16)
            av_frame = AudioFrame.from_ndarray(block.reshape(1, -1), format='s16', layout='mono')
            av_frame.sample_rate = self.audio.sampling_rate
            av_frame.pts = self.pts
            av_frame.time_base = self.audio_time_base
            self.pts += len(audio_samples)
            return av_frame

    @staticmethod
    def idle(persona, frame_index, pts=0) -> List[Frame]:
        frame = AvatarManager().get_avatar(persona).get_frame(frame_index)
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.time_base = fractions.Fraction(1, AvatarVideoDecoder.CLOCK_RATE)
        av_frame.pts = pts
        return [av_frame]

    @staticmethod
    def silence(sample_rate: int, duration_sec: float, pts: int = 0) -> List[Frame]:
        total_samples = int(sample_rate * duration_sec)
        max_samples_per_frame = int(sample_rate * AUDIO_PTIME)
        audio_time_base = fractions.Fraction(1, sample_rate)

        frames = []
        remaining = total_samples

        while remaining > 0:
            current_samples = min(remaining, max_samples_per_frame)
            block = np.zeros((1, current_samples), dtype=np.int16)

            av_frame = AudioFrame.from_ndarray(block, format='s16', layout='mono')
            av_frame.sample_rate = sample_rate
            av_frame.pts = pts
            av_frame.time_base = audio_time_base

            frames.append(av_frame)

            pts += current_samples
            remaining -= current_samples

        return frames
