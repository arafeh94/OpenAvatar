import os
import cv2
import numpy as np


def validate_path(file_path):
    parent_dir = os.path.dirname(file_path)
    if len(parent_dir) and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


class VideoWriter:
    def __init__(self, file_name, fps=24, frame_size=(640, 480)):
        self.frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(file_name, fourcc, fps, frame_size)

    def write(self, frame):
        if frame.shape[0] != self.frame_size[1] or frame.shape[1] != self.frame_size[0]:
            frame = cv2.resize(frame, self.frame_size)
        self.out.write(frame)

    def release(self):
        self.out.release()
