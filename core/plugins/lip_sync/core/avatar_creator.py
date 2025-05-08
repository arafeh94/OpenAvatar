import logging
import os
import pickle
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from tqdm import tqdm

from core.interfaces.base_face_detection import FaceDetector
from core.plugins.face_detectors import YoloFaceDetector
from core.plugins.lip_sync.wave2lip.face_detection.api import FaceAlignment, LandmarksType
from manifest import Manifest


class AvatarCreator(object):
    def __init__(self, detector: FaceDetector, **kwargs):
        self.__detector = detector
        kwargs.update(Manifest().get('avatar'))
        self.__args = SimpleNamespace(**kwargs)
        self.__logger = logging.getLogger(self.__class__.__name__)

    def create(self, persona, idle_video_path):
        self.__logger.info(f'Creating avatar for {persona}')
        self.__logger.info(f'Using {idle_video_path}')
        self.__logger.info("Capturing Video")
        video_stream = cv2.VideoCapture(idle_video_path)
        self.__logger.info("Extracting Frames")
        frames = self._extract_frames(video_stream)
        self.__logger.info("Extracting Faces")
        faces = self._extract_face(frames)
        self.__logger.info("Saving frames")
        self.__save(frames, self.__path('frame', persona))
        self.__logger.info("Saving faces")
        self.__save(faces, self.__path('face', persona))
        self.__logger.info("Saving idle video")
        shutil.copyfile(idle_video_path, self.__path('video', persona))
        self.__logger.info("Done. Results extracted to:")
        self.__logger.info("frames: {}".format(self.__path('frame', persona)))
        self.__logger.info("faces: {}".format(self.__path('face', persona)))
        self.__logger.info("video: {}".format(self.__path('video', persona)))

    # noinspection PyTypeChecker
    @staticmethod
    def __save(obj, path):
        with open(path, 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def __path(kind, persona):
        dir_path = Manifest().get('avatar')['avatar_dir']
        file_name = Manifest().get('avatar')[kind].format(persona)
        return os.path.join(dir_path, file_name)

    def _extract_frames(self, video_stream) -> list:
        frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if self.__args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // self.__args.resize_factor,
                                           frame.shape[0] // self.__args.resize_factor))

            if self.__args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = self.__args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            frames.append(frame)
        return frames

    def _extract_face(self, frames) -> list:
        predictions = [r.flat() for r in self.__detector.detect(frames, batch=self.__args.face_det_batch_size)]

        pad_y1, pad_y2, pad_x1, pad_x2 = self.__args.pads
        results = []
        for rect, image in zip(predictions, frames):
            y1 = max(0, rect[1] - pad_y1)
            y2 = min(image.shape[0], rect[3] + pad_y2)
            x1 = max(0, rect[0] - pad_x1)
            x2 = min(image.shape[1], rect[2] + pad_x2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.__args.nosmooth: boxes = self.__get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(frames, boxes)]

        return results

    @staticmethod
    def __get_smoothened_boxes(boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
