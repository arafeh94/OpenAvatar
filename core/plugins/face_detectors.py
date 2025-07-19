from typing import List, Union

import numpy as np
import torch
from ultralytics import YOLO

from core.interfaces.base_face_detection import FaceDetector, FaceResult
from manifest import Manifest


class YoloFaceDetector(FaceDetector):
    def __init__(self, **kwargs):
        weights_path = Manifest().get('yolo')['weights_path']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'mps' if torch.backends.mps.is_available() else device
        self.model = YOLO(weights_path, verbose=False)
        self.model.to(device)

    def detect(self, source, **kwargs) -> List[Union[FaceResult, List[FaceResult]]]:
        results = self.model.predict(source, **kwargs)
        detections = []
        for result in results:
            faces = result.boxes.xyxy.cpu().numpy()
            faces = [list(map(int, face)) for face in faces]
            scores = result.boxes.conf.cpu().numpy()
            if len(faces) == 0:
                detections.append(FaceResult.empty())
            for index, (face, score) in enumerate(zip(faces, scores)):
                detections.append(FaceResult(face, score, index))
        return detections

    def extract(self, frames, **kwargs):
        predictions = [r.flat() for r in self.detect(frames, **kwargs)]

        pad_y1, pad_y2, pad_x1, pad_x2 = kwargs['pads'] if 'pads' in kwargs else (0, 0, 0, 0)
        results = []
        for rect, image in zip(predictions, frames):
            if not len(rect):
                results.append([None, None, None, None])
                continue
            y1 = max(0, rect[1] - pad_y1)
            y2 = min(image.shape[0], rect[3] + pad_y2)
            x1 = max(0, rect[0] - pad_x1)
            x2 = min(image.shape[1], rect[2] + pad_x2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        # if not self.__args.nosmooth: boxes = self.__get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(frames, boxes)]

        return results
