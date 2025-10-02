import logging
import os
from logging import Logger
from pathlib import Path
from tempfile import mkdtemp
from typing import List
import numpy as np
import torch

import random, cv2

import tqdm
from decord import VideoReader
from moviepy import VideoFileClip, concatenate_audioclips

from core.plugins.face_detectors import YoloFaceDetector
from core.plugins.lip_sync.wave2lip import audio
from core.plugins.lip_sync.wave2lip.hparams import hparams
from manifest import Manifest


class DataCreator:
    def __init__(self, video_path):
        self.logger = logging.getLogger(type(self).__name__)
        self.face_detector = YoloFaceDetector()
        self.__video_path = video_path
        self.__output_dir = Manifest().query('dataset.dir')
        os.makedirs(self.__output_dir, exist_ok=True)
        self.file_list = self.__output_dir + 'file_list.txt'
        self.__video_name = '.'.join(video_path.split('/')[-1].split('.')[:-1])
        self.__video_extension = video_path.split('/')[-1].split('.')[-1]
        self.__face_path = self.__output_dir + self.__video_name + '.' + self.__video_extension
        self.__wav_path = self.__output_dir + self.__video_name + '.wav'

    def extract_audio(self):
        video = VideoFileClip(self.__face_path)
        audio = video.audio
        if audio:
            audio.write_audiofile(self.__wav_path)
            return True
        return False

    def preprocess(self, out_res=(hparams.img_size * 2, hparams.img_size * 2)):
        temp_dir = Path(mkdtemp())
        cap = cv2.VideoCapture(self.__video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_per_frame = 1 / fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        def extract_face(frame):
            # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_face, boxes = tuple(self.face_detector.extract([frame], verbose=False)[0])

            if not all(boxes):
                return None
            return frame_face

        temp_video_path = temp_dir / f"temp_faces_video_{random.randint(1000, 9999)}.mp4"
        out = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, out_res)

        selected_frame_indices = []
        self.logger.info("Filtering out empty face video frames")
        for frame_index in tqdm.tqdm(range(total_frames), desc="Filtering frames"):
            ret, frame = cap.read()
            if not ret:
                break
            face_frame = extract_face(frame)
            if face_frame is not None:
                rescaled = cv2.resize(face_frame, out_res)
                out.write(rescaled)
                selected_frame_indices.append(frame_index)

        cap.release()
        out.release()

        self.logger.info("Filtering out empty face audio frames")
        original_clip = VideoFileClip(self.__video_path)
        audio_clip = original_clip.audio
        clips = []
        for i in selected_frame_indices:
            t_start = i * duration_per_frame
            t_end = t_start + duration_per_frame
            clips.append(audio_clip.subclipped(t_start, t_end))

        filtered_audio = concatenate_audioclips(clips)

        self.logger.info("Merging audio video clips")
        video_clip = VideoFileClip(str(temp_video_path)).with_audio(filtered_audio)
        video_clip.write_videofile(self.__face_path, codec="libx264", audio_codec="aac")

        self.logger.info(f"Filtered video saved to: {self.__face_path}")

    def update_file_list(self):
        if not os.path.exists(self.file_list):
            with open(self.file_list, 'w'):
                pass
        with open(self.file_list, 'a') as f:
            f.write(self.__video_name)

    def build(self, workflow=None):
        if not workflow:
            workflow = [self.preprocess, self.extract_audio, self.update_file_list]
        for process in workflow:
            process()


class Dataset(object):
    syncnet_T = 5
    syncnet_mel_step_size = 16

    @staticmethod
    def default_list():
        data_path = Manifest().query('dataset.dir')
        list_path = f'{data_path}file_list.txt'
        with open(list_path, "r") as f:
            lines = [f"{data_path}{line.strip()}.mp4" for line in f]
        return Dataset(lines)

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

    def __len__(self):
        return len(self.videos)

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
                wav_path = ".".join(self.videos[vid_idx].split('.')[:-1] + ['wav'])
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
    # dc = DataCreator('./dataset/vid2.mp4')
    # dc.build()
    dt = Dataset.default_list()
    print(dt[0])
