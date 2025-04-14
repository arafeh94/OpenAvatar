import asyncio
import copy
import fractions, time
import threading
from queue import Queue, Empty
from typing import Optional, Callable
import numpy as np
from aiortc.contrib.media import logger
from aiortc.mediastreams import AUDIO_PTIME, MediaStreamTrack, MediaStreamError
from av import AudioFrame, VideoFrame
from av.frame import Frame
from fsspec import Callback

from core.interfaces.va import Audio
from core.tools import utils

CLOCK_RATE = 90_000
FPS = 24


class PlayerStreamTrack(MediaStreamTrack):
    def __init__(self, kind: str, idle_frame: Callable[[int], Frame] = None) -> None:
        super().__init__()
        self.kind = kind
        self._queue: asyncio.Queue[Frame] = asyncio.Queue()
        self._start: Optional[float] = None
        self.idle_frame = idle_frame
        self.pts = 0

    async def publish(self, frame: Frame):
        await self._queue.put(frame)

    async def recv(self) -> Frame:
        if self.readyState != "live":
            raise MediaStreamError

        data = await self._queue.get()
        if data is None:
            self.stop()
            raise MediaStreamError

        # data_time = float(data.pts * data.time_base)
        data_time = data.time
        self.pts = data.pts

        if self._start is None:
            self._start = time.time() - data_time
        else:
            current_time = time.time()
            wait = self._start + data_time - current_time
            # if wait is negative that means that more time pass above the data_time which was short on time
            print("processing: {}, wait: {}, start: {}, data_time: {}, time: {}"
                  .format(data, wait, self._start, data_time, current_time))
            await asyncio.sleep(wait)

        return data


class AvatarVideoContainer:
    def __init__(self, video_frames: [Frame], audio_frames: [Frame]):
        self.frames = []
        self.audio_pts = audio_frames[-1].pts
        self.video_pts = video_frames[-1].pts
        self.write(video_frames, audio_frames)

    def write(self, video_frames: [Frame], audio_frames: [Frame]):
        audio_time, video_time = 0, 0
        total_audio = float(audio_frames[-1].pts * audio_frames[-1].time_base)
        total_video = float(video_frames[-1].pts * video_frames[-1].time_base)
        i, j = 0, 0
        while i < len(video_frames) and j < len(audio_frames):
            # print("{}:{},audio_time: {}/{}, video_time: {}/{}"
            #       .format(i, j, audio_time, total_audio, video_time, total_video))
            if video_time <= audio_time:
                self.frames.append(video_frames[i])
                video_time = float(video_frames[i].pts * video_frames[i].time_base)
                i += 1
            else:
                self.frames.append(audio_frames[j])
                audio_time = float(audio_frames[j].pts * audio_frames[j].time_base)
                j += 1

        self.frames.extend(video_frames[i:])
        self.frames.extend(audio_frames[j:])
        # Video frame correction to avoid desynchronization.
        # Works when audio have more frames of total less than 1/fps(s)
        # if len(audio_frames[j:]) > 0:
        #     pts_increment = (sum([f.samples for f in audio_frames[j:]]) / audio_frames[-1].sample_rate) * CLOCK_RATE
        #     np_frames = pts_increment * FPS / CLOCK_RATE
        #     frames_ts = utils.split_ones(np_frames) * (CLOCK_RATE / FPS)
        #     latest_v_frame = video_frames[-1]
        #     for frame_ts in frames_ts:
        #         duplicate_frame = video_frames[-1].to_ndarray(format='rgb24')
        #         duplicate_frame = VideoFrame.from_ndarray(duplicate_frame, format="bgr24")
        #         duplicate_frame.pts = latest_v_frame.pts + frame_ts
        #         self.frames.append(duplicate_frame)
        #         latest_v_frame = duplicate_frame
        #     self.frames.extend(audio_frames[j:])

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.frames):
            raise StopIteration
        frame = self.frames[self._index]
        self._index += 1
        return frame


def video_frames_decoder(frame_buffer, initial_pts=0):
    av_frames = []
    pts = 0
    pts_increment = int(CLOCK_RATE / FPS)

    for i, frame in enumerate(frame_buffer):
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts + initial_pts
        av_frame.time_base = fractions.Fraction(1, CLOCK_RATE)
        pts += pts_increment
        av_frames.append(av_frame)
    return av_frames


def audio_frames_decoder(audio: Audio, initial_pts=0):
    audio_time_base = fractions.Fraction(1, audio.sampling_rate)
    av_frames = []
    pts = 0
    batches = utils.create_batches(audio.samples, int(audio.sampling_rate * AUDIO_PTIME))
    for i, audio_samples in enumerate(batches):
        block = (np.array(audio_samples) / max(1, np.max(np.abs(audio_samples))) * 32767).astype(np.int16)
        av_frame = AudioFrame.from_ndarray(block.reshape(1, -1), format='s16', layout='mono')
        av_frame.sample_rate = audio.sampling_rate
        av_frame.pts = pts + initial_pts
        av_frame.time_base = audio_time_base
        av_frames.append(av_frame)
        pts += len(audio_samples)
    return av_frames


def avatar_worker_decode(
        loop,
        buffer_queue: Queue,
        video_track: PlayerStreamTrack,
        audio_track: PlayerStreamTrack,
        quit_event
):
    while not quit_event.is_set():
        buffer = buffer_queue.get()
        audio_pts_start, video_pts_start = 0, 0
        while True:
            try:
                video, audio, text = next(buffer)
                audio_frames = audio_frames_decoder(audio, initial_pts=audio_pts_start)
                video_frames = video_frames_decoder(video, initial_pts=video_pts_start)
                container = AvatarVideoContainer(video_frames, audio_frames)
                audio_pts_start, video_pts_start = container.audio_pts, container.video_pts
                for frame in container:
                    if isinstance(frame, VideoFrame):
                        asyncio.run_coroutine_threadsafe(video_track.publish(frame), loop)
                    else:
                        asyncio.run_coroutine_threadsafe(audio_track.publish(frame), loop)
            except StopIteration:
                avatar_worker_decode(loop, buffer_queue, video_track, audio_track, quit_event)


class AvatarMediaPlayer:
    def __init__(self) -> None:
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None
        self.__audio: PlayerStreamTrack = PlayerStreamTrack("audio")
        self.__video: PlayerStreamTrack = PlayerStreamTrack("video")
        self.__buffer_queue: Queue[Frame] = Queue()

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance.
        """
        return self.__video

    def start(self, buffer) -> None:
        self.__buffer_queue.put(buffer)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="avatar-media-player",
                target=avatar_worker_decode,
                args=(
                    asyncio.get_event_loop(),
                    self.__buffer_queue,
                    self.__video,
                    self.__audio,
                    self.__thread_quit,
                ),
            )
            self.__thread.start()

    def stop(self) -> None:
        if self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

    def __log_debug(self, msg: str, *args) -> None:
        logger.debug(f"MediaPlayer(%s) {msg}", 'Avatar Container', *args)


class MediaSink:
    def __init__(self, track: MediaStreamTrack):
        self.track = track
        self._stop_event = asyncio.Event()

    async def start(self):
        try:
            while not self._stop_event.is_set():
                frame = await self.track.recv()
                self._process_frame(frame)

        except MediaStreamError:
            print("Error: Stream ended or invalid data.")
        finally:
            print("MediaSink stopped.")

    def stop(self):
        self._stop_event.set()

    def _process_frame(self, frame: Frame):
        # Example frame processing: just print the timestamp of the frame
        print(f"Processing frame {frame} with timestamp: {frame.pts}")
