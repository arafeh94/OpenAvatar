import logging
from enum import Enum

import aiohttp
from livekit import rtc

API_CONFIG = {
    "serverUrl": "https://api.heygen.com",
    "token": "YjQzZDJiNmNkMTQ2NGVkNTkyNzcwY2NmMTgyMWQ4MzItMTczMjIwMjU1Nw==",
}


class Heygen:
    def __init__(self, avatar_id):
        self.logger = logging.getLogger("Heygen")
        self.video_track = None
        self.audio_track = None
        self.status = None
        self.room = None
        self.avatarId = avatar_id
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_CONFIG['token']}"
        }
        self.session_info = None

    async def close_session(self):
        async with aiohttp.ClientSession() as session:
            await session.post(
                url=f"{API_CONFIG['serverUrl']}/v1/streaming.stop",
                headers=self.headers,
                json={"session_id": self.session_info["session_id"]}
            )

        if self.room:
            await self.room.disconnect()

        self.sessionInfo = None
        self.room = None

    async def interrupt(self):
        async with aiohttp.ClientSession() as session:
            await session.post(
                url=f"{API_CONFIG['serverUrl']}/v1/streaming.interrupt",
                headers=self.headers,
                json={"session_id": self.session_info["session_id"]}
            )

    async def create_session(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{API_CONFIG['serverUrl']}/v1/streaming.new",
                    json={"version": "v2", "avatar_id": self.avatarId},
                    headers=self.headers
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Failed to create session: {resp.status} - {text}")
                self.session_info = (await resp.json())["data"]
                logging.info("Session info:", self.session_info)

            async with session.post(
                    f"{API_CONFIG['serverUrl']}/v1/streaming.start",
                    json={"session_id": self.session_info["session_id"]},
                    headers=self.headers
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Failed to start session: {resp.status} - {text}")

        room = rtc.Room()

        @room.on("connected")
        def on_connected():
            self.status = 'connected'

        @room.on("disconnected")
        def on_connected():
            self.status = 'disconnected'

        @room.on("track_subscribed")
        def on_track(track, publication, participant):
            self.logger.info(f"Track received: {track.kind}")
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self.video_track = rtc.VideoStream(track)
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                self.audio_track = rtc.AudioStream(track)

        await room.connect(
            url=self.session_info["url"],
            token=self.session_info["access_token"]
        )

        self.room = room

    def time_take_lot(self):
        return 1

    async def repeat(self, text: str, type='repeat'):
        url = f"{API_CONFIG['serverUrl']}/v1/streaming.task"
        data = {
            "session_id": self.session_info["session_id"],
            "text": text,
            "task_type": type
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=data) as response:
                return response.status == 200

    def is_ready(self):
        return self.video_track is not None and self.audio_track is not None
