from torch.backends.mkl import verbose
from ultralytics import YOLO
import torch

from core.plugins.face_detectors import YoloFaceDetector
from core.plugins.lip_sync.core.avatar import Avatar
from core.plugins.lip_sync.core.models import AvatarWave2LipModel


def main():
    avatar = Avatar('az', AvatarWave2LipModel())
    avatar.init()
    face_detector = YoloFaceDetector()
    results = face_detector.detect(avatar.video_frames[0:10], verbose=True)
    print(results)


if __name__ == '__main__':
    main()
