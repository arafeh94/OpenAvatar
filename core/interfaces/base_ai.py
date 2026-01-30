import queue
import threading
from abc import ABC, abstractmethod
import concurrent.futures

import torch


class AIModel(ABC):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls, *args, **kwargs)
        return cls._instances[cls]

    def __init__(self):
        self.model = self.load_model()
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def load_model(self):
        pass

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            return self.model(*args, **kwargs)


class NonBlockingAIModel(AIModel, ABC):
    def __init__(self):
        super().__init__()
        self.model.to(self.device)
        self.task_queue = queue.Queue()
        self.executor_thread = threading.Thread(target=self._process_queue)
        self.executor_thread.daemon = True
        self.executor_thread.start()

    def _inference_task(self, *args, **kwargs):
        """The method that actually performs the inference."""
        with torch.no_grad():
            return self.model(*args, **kwargs)

    def _process_queue(self):
        while True:
            future, args, kwargs = self.task_queue.get()
            if future.cancelled():
                continue
            try:
                result = self._inference_task(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self.task_queue.task_done()

    def __call__(self, *args, **kwargs):
        """
        Enqueue inference task to a background queue and return an awaitable future.
        """
        future = concurrent.futures.Future()
        self.task_queue.put((future, args, kwargs))
        return future

    def shutdown(self):
        """Shutdown the executor thread."""
        self.executor_thread.join()
