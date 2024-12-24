from abc import ABC, abstractmethod


class Pipe(ABC):
    @abstractmethod
    def exec(self, obj: any, flow: []) -> any:
        pass
