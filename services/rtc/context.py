from core.plugins.lip_sync.core.avatar import AvatarManager
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech


class AppContext:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppContext, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    # noinspection PyUnresolvedReferences
    def __init__(self):
        if not self._initialized:
            self.peers: dict[str, 'ServerPeer'] = {}
            self.avatar_manager = AvatarManager(AvatarWave2LipModel(), MicrosoftText2Speech())
            self._initialized = True


