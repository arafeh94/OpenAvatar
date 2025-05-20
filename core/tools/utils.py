import importlib
import io
import logging
import threading
import typing

import numpy as np


class ObservableEvent:

    def __init__(self):
        self._event = threading.Event()
        self._listeners = []
        self._lock = threading.Lock()

    def add_listener(self, callback: typing.Callable, args: tuple = None):
        with self._lock:
            self._listeners.append((callback, args))

    def set(self):
        if self._event.is_set(): return
        self._event.set()
        for callback in self._listeners:
            fn, args = callback
            threading.Thread(target=fn, args=args).start()

    def wait(self, timeout=None):
        return self._event.wait(timeout)

    def is_set(self):
        return self._event.is_set()

    def clear(self):
        self._event.clear()


class SafeValue:
    def __init__(self, initial_value=None):
        self._lock = threading.Lock()
        self._value = initial_value

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, new_value):
        with self._lock:
            self._value = new_value


def get_class_instance(class_path: str, *args, **kwargs):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ref = getattr(module, class_name)
    return class_ref(*args, **kwargs)


def enable_logging(file_name=None, level=logging.INFO):
    if file_name:
        logging.basicConfig(filename=file_name, filemode='w', datefmt='%H:%M:%S', level=level)
    else:
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level)


def info(logger, content, **args):
    if 'log' in args and args['log']:
        logger.info(content)


def split_ones(arr, precision=10):
    whole = int(arr)
    fraction = round(arr - whole, precision)
    arr = [1] * whole
    if fraction > 0:
        arr.append(fraction)
    return np.array(arr)


def create_batches(arr, batch_size):
    num_batches = len(arr) // batch_size + (1 if len(arr) % batch_size != 0 else 0)
    batches = [arr[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    return batches


def sec(t):
    millis = round((t - int(t)) * 1000, 4)
    return f"{millis:.4f}"


def as_bytes(generator):
    data = io.BytesIO()
    for chunk in generator:
        if chunk:
            data.write(chunk)
    data.seek(0)
    return data
