from abc import ABC, abstractmethod

from core.interfaces.va import VoiceConvertable


class Text2Speech(ABC):
    @abstractmethod
    def convert(self, text, **kwargs) -> VoiceConvertable:
        pass
