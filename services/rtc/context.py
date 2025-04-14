from collections import defaultdict
from typing import Dict, Any

from core.plugins.lip_sync.core.avatar_extentions import AvatarManager
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech


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
            self.avatar_manager = AvatarManager()
            self.peer_preferences: Dict[str, Dict[str, Any]] = defaultdict(dict)
            self._initialized = True

    def idle_frame(self, peer_id):
        peer_preferences = self.peer_preferences[peer_id]
        if "idle_frame_index" not in peer_preferences:
            peer_preferences["idle_frame_index"] = 0
        idle_frame = peer_preferences["idle_frame_index"]
        peer_preferences["idle_frame_index"] += 1
        return idle_frame
