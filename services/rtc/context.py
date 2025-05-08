from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from core.plugins.lip_sync.core.avatar_extentions import AvatarManager
from core.plugins.rag.chat import ChatService
from services.rtc.src.typing import ServerPeer


class AppContext:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppContext, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.peers = {}
            self.chat = ChatService.create(model_name='gpt-4o-mini', max_tokens=16384)
            self.avatar_manager = AvatarManager()
            self.peer_preferences: Dict[str, Dict[str, Any]] = defaultdict(dict)
            self.__executor = ThreadPoolExecutor(max_workers=10)
            self._initialized = True

    def add_peer(self, peer_id, peer):
        self.peers[peer_id] = peer

    def del_peer(self, peer_id):
        del self.peers[peer_id]

    def peer(self, token) -> 'ServerPeer':
        return self.peers[token]

    def idle_frame(self, peer_id):
        peer_preferences = self.peer_preferences[peer_id]
        if "idle_frame_index" not in peer_preferences:
            peer_preferences["idle_frame_index"] = 0
        idle_frame = peer_preferences["idle_frame_index"]
        peer_preferences["idle_frame_index"] += 1
        return idle_frame

    def run_in_thread(self, fn, on_end, *args, **kwargs):
        future = self.__executor.submit(fn, *args, **kwargs)

        def callback(fut):
            try:
                result = fut.result()
                on_end(result)
            except Exception as e:
                print(f"Exception in thread: {e}")

        future.add_done_callback(callback)
