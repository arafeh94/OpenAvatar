from abc import ABC, abstractmethod
from typing import List, Union


class FaceResult:
    def __init__(self, bbox, score, index=0):
        self.bbox = bbox
        self.score = score
        self.index = index

    def flat(self):
        return *self.bbox,

    def __repr__(self):
        return "bbox: " + str(self.bbox) + " score: " + str(self.score) + " index:" + str(self.index)

    def is_empty(self):
        return self.bbox == []

    @staticmethod
    def empty():
        return FaceResult([], 0, 0)


class FaceDetector(ABC):

    @abstractmethod
    def detect(self, source, **kwargs) -> List[Union[FaceResult, List[FaceResult]]]: ...


class FakesFaceDetector(FaceDetector):
    def detect(self, source, **kwargs):
        return [FaceResult([0, 0, 100, 100], 1)]


if __name__ == "__main__":
    face_detection = FakesFaceDetector()
    res = face_detection.detect(['test'])
    print(res, res[0].flat())
