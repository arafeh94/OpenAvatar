import abc
import logging


class SetupPhase(abc.ABC):
    def __init__(self):
        self.logger = logging.getLogger(SetupPhase.__name__)

    @abc.abstractmethod
    def exec(self):
        pass

    @abc.abstractmethod
    def description(self):
        pass
