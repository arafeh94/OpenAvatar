import queue
import threading
import time


class NonBlockingLookaheadGenerator:
    def __init__(self, gen):
        self._gen = gen
        self._queue = queue.Queue(maxsize=1)
        self._done = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._prefetch, daemon=True)
        self._prefetch()

    def _prefetch(self):
        """Fetch the next value and put it into the queue."""
        try:
            next_value = next(self._gen)
            self._queue.put(next_value)
        except StopIteration:
            self._queue.put(None)
        except Exception as e:
            self._queue.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next value from the queue, and prefetch the next one in a non-blocking way."""
        if self._done:
            raise StopIteration

        result = self._queue.get()

        if isinstance(result, Exception):
            raise result
        if result is None:
            self._done = True
            raise StopIteration

        if self._queue.empty() and not self._done:
            self._thread = threading.Thread(target=self._prefetch, daemon=True)
            self._thread.start()

        return result

    def stop(self):
        """Stop the generator."""
        with self._lock:
            if self._thread.is_alive():
                self._done = True
                self._queue.put(None)
                self._thread.join()
