import logging

from external.core.pipeline import Pipe


class InputPrompt(Pipe):
    def __init__(self, msg=None):
        super().__init__()
        self.msg = msg

    def exec(self, arg, flow):
        return input(self.msg)


class WaitPrompt(Pipe):
    def exec(self, arg, flow) -> any:
        input('Press Enter continue...')
        return arg


class TimeingFileLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TimeingFileLogger, cls).__new__(cls)
            cls._instance._logger = logging.getLogger('singleton_logger')
            cls._instance._logger.setLevel(logging.DEBUG)
            file_handler = logging.FileHandler('timings.log')
            file_handler.setLevel(logging.DEBUG)
            cls._instance._logger.addHandler(file_handler)
        return cls._instance

    def get_logger(self):
        return self._logger


class ETLogger(Pipe):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.log_handler = TimeingFileLogger()

    def exec(self, arg, flow) -> any:
        self.log_handler.get_logger().critical(json.dumps(self.pipeline.execution_times()))
        return arg


class AsString(Pipe):
    def exec(self, arg, flow) -> any:
        return str(arg)


class Mirror(Pipe):
    def __init__(self, prefix=None):
        self.prefix = prefix if prefix else ''

    def exec(self, arg, flow):
        print(self.prefix, arg)
        return arg
