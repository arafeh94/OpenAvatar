import threading


class AtomicID:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, start=0):
        if not hasattr(self, "_initialized"):
            self._lock = threading.Lock()
            self._id = start
            self._initialized = True

    def fetch(self):
        with self._lock:
            self._id += 1
            return self._id
