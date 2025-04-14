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
from core.plugins.lip_sync.core.avatar_extentions import AvatarVideoDecoder
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

    def _next_pts(self, frame) -> int:
        if self.kind == "audio":
            pts = int(self.pts + frame.samples)
        else:
            pts = int(self.pts + CLOCK_RATE / FPS)
        return pts

    async def publish(self, frame: Frame):
        await self._queue.put(frame)

    async def recv(self) -> Frame:
        if self.readyState != "live":
            raise MediaStreamError

        data = await self._queue.get()
        if data is None:
            self.stop()
            raise MediaStreamError

        data.pts = self._next_pts(data)

        data_time = data.time
        self.pts = data.pts

        if self._start is None:
            self._start = time.time() - data_time
        else:
            current_time = time.time()
            wait = self._start + data_time - current_time
            await asyncio.sleep(wait)

        return data


def avatar_worker_decode(
        loop,
        buffer_queue: Queue,
        video_track: PlayerStreamTrack,
        audio_track: PlayerStreamTrack,
        quit_event
):
    while not quit_event.is_set():
        buffer = buffer_queue.get()
        while True:
            try:
                video, audio, text = next(buffer)
                container = AvatarVideoDecoder.decode((video, audio))
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
    """
    A media blackhole to consume the media player recv.
    Examples:
        media_player = AvatarMediaPlayer()
        media_player.start(buffer)
        await MediaSink(media_player.video, media_player.audio).start()
    """
    def __init__(self, *tracks: [MediaStreamTrack]):
        self.tracks = tracks
        self._stop_event = asyncio.Event()

    async def start(self):
        try:
            while not self._stop_event.is_set():
                for track in self.tracks:
                    frame = await track.recv()
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
