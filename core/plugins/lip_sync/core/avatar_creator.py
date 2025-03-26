from types import SimpleNamespace

import cv2
import numpy as np
import torch
from tqdm import tqdm

from core.plugins.lip_sync.wave2lip.face_detection.api import FaceAlignment, LandmarksType


class AvatarCreator(object):
    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_args = {
            'face_det_batch_size': 1
        }
        self.args = SimpleNamespace(**kwargs)

    # noinspection PyPep8Naming
    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def get_face_frame(self, frames, auto=True):
        if auto:
            return self.face_detect(frames)
        else:
            y1, y2, x1, x2 = self.args.box
            results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
            return results

    def face_detect(self, frames):
        # noinspection PyProtectedMember
        detector = FaceAlignment(LandmarksType._2D, flip_input=False, device=self.device)

        batch_size = self.args.face_det_batch_size

        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(frames), batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(frames[i:i + batch_size])))

            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.args.pads
        fnr, x1, y1, x2, y2 = 0, 0, 0, 100, 100
        for rect, image in zip(predictions, frames):
            if rect is None:
                # TODO: check this one, internal created path
                cv2.imwrite('avatar/temp/faulty_frame.jpg', image)
                print(f'Face not detected in {fnr}! Ensure the video contains a face in all the frames.')
            else:
                y1 = max(0, rect[1] - pady1)
                y2 = min(image.shape[0], rect[3] + pady2)
                x1 = max(0, rect[0] - padx1)
                x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])
            fnr = fnr + 1

        boxes = np.array(results)
        if not self.args.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(frames, boxes)]

        del detector
        return results
