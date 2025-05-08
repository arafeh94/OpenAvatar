from core.plugins.face_detectors import YoloFaceDetector
from core.plugins.lip_sync.core.avatar_creator import AvatarCreator
from core.tools import utils

utils.enable_logging(level='INFO')
creator = AvatarCreator(YoloFaceDetector())
creator.create('azzam', 'az2.mp4')
