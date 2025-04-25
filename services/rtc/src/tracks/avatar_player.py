import asyncio
import logging
import time
import threading
from asyncio import QueueEmpty
from queue import Queue, Empty
from typing import Optional, Callable, List
from aiortc.contrib.media import logger
from aiortc.mediastreams import MediaStreamTrack, MediaStreamError, AUDIO_PTIME
from av import VideoFrame
from av.frame import Frame

from core.plugins.lip_sync.core.avatar_extentions import AvatarVideoDecoder
from manifest import Manifest
from services.rtc.context import AppContext


class PlayerStreamTrack(MediaStreamTrack):
    def __init__(self, kind: str, idle_frame: Callable[[], Frame]) -> None:
        super().__init__()
        self.kind = kind
        self._queue: asyncio.Queue[Frame] = asyncio.Queue()
        self._start: Optional[float] = None
        self.idle_frame = idle_frame
        self.pts = 0
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{self.kind}")
        self.__log = Manifest().query('rtc.behaviors.log_timestamps', False)
        # self.__log = True

    def _next_pts(self, frame) -> int:
        if self.kind == "audio":
            pts = int(self.pts + frame.samples)
        else:
            pts = int(self.pts + AvatarVideoDecoder.CLOCK_RATE / AvatarVideoDecoder.FPS)
        return pts

    async def publish(self, frame: Frame):
        await self._queue.put(frame)

    async def recv(self) -> Frame:
        if self.readyState != "live":
            raise MediaStreamError
        try:
            data = self._queue.get_nowait()
            frame_type = 'lip-sync'
        except QueueEmpty:
            data = self.idle_frame()
            frame_type = 'idle'

        if data is None:
            self.stop()
            raise MediaStreamError

        data.pts = self._next_pts(data)

        data_time = data.time
        self.pts = data.pts

        current_time = time.time()

        if self._start is None:
            self._start = current_time
            self.__log and self.logger.info(
                f"frame_type:{frame_type:<8},\t"
                f"start_time:{self._start:.4f},\t"
                f"current_time:{current_time:.4f},\t"
                f"data_time:{round(data_time, 4):.4f}s,\t"
                f"wait: {round(0, 4):+.4f}s,\t"
                f"data: {data}"
            )
        else:
            wait = self._start + data_time - current_time
            self.__log and self.logger.info(
                f"frame_type:{frame_type:<8}\t"
                f"start_time:{self._start:.4f},\t"
                f"current_time:{current_time:.4f},\t"
                f"data_time:{round(data_time, 4):.4f}s,\t"
                f"wait: {round(wait, 4):+.4f}s,\t"
                f"{data}"
            )
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
                publish(video, audio, video_track, audio_track, loop)
            except StopIteration:
                break


def publish(video, audio, video_track, audio_track, loop):
    AVD = AvatarVideoDecoder

    def publisher(_frame):
        if isinstance(_frame, VideoFrame):
            asyncio.run_coroutine_threadsafe(video_track.publish(_frame), loop)
        else:
            asyncio.run_coroutine_threadsafe(audio_track.publish(_frame), loop)

    AVD.decode((video, audio), publisher)


class IdleFrames:
    @staticmethod
    def audio() -> Callable[[], Frame]:
        def inner() -> Frame:
            return AvatarVideoDecoder.silence(16000, AUDIO_PTIME)[0]

        return inner

    @staticmethod
    def video(peer_id, persona) -> Callable[[], Frame]:
        def inner() -> Frame:
            return AvatarVideoDecoder.idle(persona, AppContext().idle_frame(peer_id))[0]

        return inner


class AvatarMediaPlayer:
    def __init__(self, peer_id, persona) -> None:
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None
        self.__audio: PlayerStreamTrack = PlayerStreamTrack("audio", IdleFrames.audio())
        self.__video: PlayerStreamTrack = PlayerStreamTrack("video", IdleFrames.video(peer_id, persona))
        self.__buffer_queue: Queue[Frame] = Queue()
        self.__persona = persona
        self.__peer_id = peer_id

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
        if buffer:
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
                daemon=True,
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

    def __init__(self, *tracks: MediaStreamTrack):
        self.__tracks = tracks
        self._stop_event = asyncio.Event()

    async def start(self):
        try:
            while not self._stop_event.is_set():
                for track in self.__tracks:
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
