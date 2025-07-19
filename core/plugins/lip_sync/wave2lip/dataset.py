import logging
import os
from pathlib import Path
from tempfile import mkdtemp
from typing import List
import numpy as np
import torch

import random, cv2

import tqdm
from decord import VideoReader
from librosa.util.files import index
from matplotlib import pyplot as plt
from moviepy import VideoFileClip, concatenate_audioclips, AudioFileClip
from torch.backends.mkl import verbose

from core.interfaces.base_face_detection import FaceResult
from core.plugins.face_detectors import YoloFaceDetector
from core.plugins.lip_sync.wave2lip import audio
from core.plugins.lip_sync.wave2lip.hparams import hparams
from core.tools import utils


class DataCreator:
    def __init__(self, video_path):
        self.__video_path = video_path
        self.__output_dir = './dataset/'
        self.__video_name = '.'.join(video_path.split('/')[-1].split('.')[:-1])
        self.__video_extension = video_path.split('/')[-1].split('.')[-1]
        self.__clean_path = self.__output_dir + self.__video_name + '_clean.' + self.__video_extension
        self.__face_path = self.__output_dir + self.__video_name + '_face.' + self.__video_extension
        self.__wav_path = self.__output_dir + self.__video_name + '.wav'

    def extract_face(self, res=(hparams.img_size, hparams.img_size), batch_size=30):
        face_detector = YoloFaceDetector()
        vr = VideoReader(self.__video_path)
        fps = vr.get_avg_fps()
        height, width = res
        out = cv2.VideoWriter(self.__face_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
        for start in tqdm.tqdm(range(0, len(vr), batch_size), desc="Face extraction"):
            end = min(start + batch_size, len(vr))
            frame_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in vr[start:end].asnumpy()]
            frame_face_bgr = face_detector.extract(frame_bgr, verbose=False)
            for index, face_frame in enumerate(frame_face_bgr):
                if not all(face_frame[1]):
                    raise Exception(f"A frame with no face detected at pos {index} between frames {start} and {end}. "
                                    f"Clean the video first.")
                rescaled = cv2.resize(face_frame[0], (width, height))
                out.write(rescaled)
        out.release()
        return True

    def extract_audio(self):
        video = VideoFileClip(self.__video_path)
        audio = video.audio
        if audio:
            audio.write_audiofile(self.__wav_path)
            return True
        return False

    def clean(self):
        face_detector = YoloFaceDetector()
        temp_dir = Path(mkdtemp())
        cap = cv2.VideoCapture(self.__video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_per_frame = 1 / fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger = logging.getLogger('__clean__')

        def has_face(frame):
            is_detected = not face_detector.detect(frame, verbose=False)[0].is_empty()
            return is_detected

        temp_video_path = temp_dir / "temp_faces_video.mp4"
        out = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        selected_frame_indices = []
        logger.info("Filtering out empty face video frames")
        for frame_index in tqdm.tqdm(range(total_frames), desc="Filtering frames"):
            ret, frame = cap.read()
            if not ret:
                break
            if has_face(frame):
                out.write(frame)
                selected_frame_indices.append(frame_index)

        cap.release()
        out.release()

        logger.info("Filtering out empty face audio frames")
        original_clip = VideoFileClip(self.__video_path)
        audio_clip = original_clip.audio
        clips = []
        for i in selected_frame_indices:
            t_start = i * duration_per_frame
            t_end = t_start + duration_per_frame
            clips.append(audio_clip.subclipped(t_start, t_end))

        filtered_audio = concatenate_audioclips(clips)

        logger.info("Merging audio video clips")
        video_clip = VideoFileClip(str(temp_video_path)).with_audio(filtered_audio)
        video_clip.write_videofile(self.__clean_path, codec="libx264", audio_codec="aac")

        logger.info(f"Filtered video saved to: {self.__clean_path}")


class Dataset(object):
    syncnet_T = 5
    syncnet_mel_step_size = 16

    def __init__(self, videos: List[str]):
        self.videos = videos
        self.loaded = {}
        self.loaded_mels = {}

    def get_decoder(self, vid_index):
        if vid_index not in self.loaded:
            self.loaded[vid_index] = VideoReader(self.videos[vid_index])
        return self.loaded[vid_index]

    def get_window(self, video_index, start_index):
        return self.get_decoder(video_index)[start_index:start_index + Dataset.syncnet_T]

    def as_img(self, decord_video_frames):
        window = []
        for frame in decord_video_frames.asnumpy():
            try:
                img = cv2.resize(frame, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)
        return window

    def crop_audio_window(self, spec, frame_index):
        start_idx = int(80. * (frame_index / float(hparams.fps)))

        end_idx = start_idx + Dataset.syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def get_segmented_mels(self, spec, frame_index):
        mels = []
        assert self.syncnet_T == 5
        if frame_index - 2 < 0: return None
        for i in range(frame_index, frame_index + self.syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != self.syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __getitem__(self, idx):
        while 1:
            vid_idx = random.randint(0, len(self.videos) - 1)
            video = self.get_decoder(vid_idx)
            start_frame = random.randint(0, len(video) - 1)
            wrong_start_frame = random.randint(0, len(video) - 1)
            while start_frame == wrong_start_frame:
                wrong_start_frame = random.randint(0, len(video) - 1)

            video_frames = self.get_window(vid_idx, start_frame)
            wrong_video_frames = self.get_window(vid_idx, wrong_start_frame)
            if video_frames is None or wrong_video_frames is None:
                continue

            window = self.as_img(video_frames)
            if window is None:
                continue

            wrong_window = self.as_img(wrong_video_frames)
            if wrong_window is None:
                continue

            try:
                wav_path = ".".join(dt.videos[0].split('.')[:-1] + ['wav'])
                wav = audio.load_wav(wav_path, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), start_frame)

            if mel.shape[0] != self.syncnet_mel_step_size:
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), start_frame)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2] // 2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y


if __name__ == '__main__':
    dc = DataCreator('./dataset/vid2.mp4')
    dc.clean()
