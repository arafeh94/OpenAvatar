from typing import List, Union

import torch
from ultralytics import YOLO

from core.interfaces.base_face_detection import FaceDetector, FaceResult
from manifest import Manifest


class YoloFaceDetector(FaceDetector):
    def __init__(self, **kwargs):
        weights_path = Manifest().get('yolo')['weights_path']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'mps' if torch.backends.mps.is_available() else device
        self.model = YOLO(weights_path, **kwargs)
        self.model.to(device)

    def detect(self, source, **kwargs) -> List[Union[FaceResult, List[FaceResult]]]:
        results = self.model.predict(source, **kwargs)
        detections = []
        for result in results:
            faces = result.boxes.xyxy.cpu().numpy()
            faces = [list(map(int, face)) for face in faces]
            scores = result.boxes.conf.cpu().numpy()
            for index, (face, score) in enumerate(zip(faces, scores)):
                detections.append(FaceResult(face, score, index))
        if len(detections) == 1:
            return detections[0]
        return detections
