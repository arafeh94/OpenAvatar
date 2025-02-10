from collections import UserDict
from time import time


class ExpiringDict(UserDict):

    def __init__(self, expiration_time=3600, cleanup_interval=10):
        super().__init__()
        self.expiration_time = expiration_time
        self.cleanup_interval = cleanup_interval
        self.access_times = {}
        self.access_count = 0

    def __setitem__(self, key, value):
        self.try_clean()
        self.data[key] = value
        self.access_times[key] = time()

    def __getitem__(self, key):
        self.try_clean()
        if key in self.data:
            self.access_times[key] = time()
            return self.data[key]
        else:
            raise KeyError(f"Key '{key}' not found.")

    def __delitem__(self, key):
        if key in self.data:
            del self.data[key]
            del self.access_times[key]

    def try_clean(self):
        self.access_count += 1
        if self.access_count >= self.cleanup_interval:
            self._cleanup()
            self.access_count = 0

    def _cleanup(self):
        expired_keys = [key for key, last_access in self.access_times.items()
                        if time() - last_access >= self.expiration_time]
        for key in expired_keys:
            self.__delitem__(key)
