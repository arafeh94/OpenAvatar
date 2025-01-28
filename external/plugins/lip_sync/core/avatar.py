import logging
import os
import pickle
import platform
import subprocess
from logging import Logger
from types import SimpleNamespace
import time
import random

import numpy as np
import torch
import librosa
from moviepy import AudioFileClip, ImageSequenceClip

from external.tools import memory_profiler
from manifest import Manifest
from external.plugins.lip_sync.wave2lip import audio
from external.core.interfaces.base_ai import AIModel
from external.tools.async_generator import NonBlockingLookaheadGenerator
from external.tools.atomic_id import AtomicID
import cv2


class Avatar(object):
    """
   Avatar class for generating lip-sync animations based on video and audio input.

   Parameters
   ----------
   avatar_id : str
       A unique identifier for the avatar.
   lip_model : AIModel
       The AI model used for generating the lip-sync animation.
       This should be an instance of a model that handles lip-sync generation based on audio input.
   kwargs : dict
       A dictionary containing configuration options for lip-sync generation.
       Default values are taken from the Manifest. The dictionary keys are:

       - avatar_dir (str): The directory path where avatar-related assets are stored.
       - frame (str): A file path template for frames. Each frame is generated based on this template.
         TODO: This should be updated based on further specifications for frame generation.
       - face (str): A file path template for face data (e.g., facial features or landmarks).
         TODO: This should be updated based on further specifications for face data storage.
       - video (str): A file path template for video data.
         TODO: This should be updated based on further specifications for video processing.
       - fps (int): The frame rate (frames per second) used for generating lip-sync.
       - wav2lip_batch_size (int): The batch size used in the Wav2Lip model.
         This should be divisible by `fps` for best results.
         If not divisible, the output may exhibit cuts or asynchronization between audio and video.
       - mel_step_size (int): The step size for the mel spectrogram.
       - pads (list): A list specifying padding values [top, bottom, left, right] for the cropping box.
       - face_det_batch_size (int): The batch size used for face detection.
       - resize_factor (int): The factor by which to resize the input video.
       - crop (list): A list specifying the cropping box [top, bottom, left, right] for the video.
       - box (list): A list defining the bounding box for face detection.
       - rotate (bool): Whether to rotate the face for correct alignment.
       - nosmooth (bool): If set to True, disables smoothing of the face animation.
       - img_size (int): The size of the image used in the Wav2Lip model.
       - static (bool): If set to True, lip-sync is generated from an impact, without using face detection.
         If False, the system uses automatic face detection for synchronization.

   avatar = Avatar(avatar_id="avatar123", lip_model=some_lip_model, **kwargs)

   Notes
   -----
   - Ensure that `wav2lip_batch_size` is divisible by `fps` for optimal synchronization of lip-sync output.
   - When `static` is set to `True`, lip-sync generation is driven by an impact (e.g., a predefined sequence),
     while `False` triggers automatic face detection for dynamic adjustments.
   """

    def __init__(self, avatar_id, lip_model: AIModel, **kwargs):
        self.logger = logging.getLogger("Avatar")
        self.avatar_id = avatar_id
        self.lip_model = lip_model

        default_values = Manifest().get('avatar')
        self.args = SimpleNamespace(**{**dict(default_values), **kwargs})

        # initialization
        self.face_detection_results = None
        self.video_frames = None
        self.id = AtomicID().fetch()

    def init(self):
        self.face_detection_results = self._get_avatar_face_detection_results()
        self.video_frames = self._get_avatar_videos_frames()

    def _get_avatar_videos_frames(self):
        if self.video_frames is not None:
            return self.video_frames

        avatar_video_frames_path = os.path.join(self.args.avatar_dir, self.args.frame.format(self.avatar_id))
        if not os.path.exists(avatar_video_frames_path):
            raise Exception("avatar_video_frames_path does not exists. path: {}".format(avatar_video_frames_path))

        with open(avatar_video_frames_path, 'rb') as f:
            full_frames = pickle.load(f)
            return full_frames

    def _get_avatar_face_detection_results(self):
        if self.face_detection_results is not None:
            return self.face_detection_results

        avatar_face_det_path = os.path.join(self.args.avatar_dir, self.args.face.format(self.avatar_id))
        if not os.path.exists(avatar_face_det_path):
            raise Exception("avatar_video_frames_path does not exists. path: {}".format(avatar_face_det_path))

        with open(avatar_face_det_path, 'rb') as f:
            face_det_results = pickle.load(f)
            return face_det_results

    def _get_avatar_video(self):
        avatar_video_path = os.path.join(self.args.avatar_dir, self.args.video.format(self.avatar_id))

        if not os.path.exists(avatar_video_path):
            raise Exception("avatar_video_frames_path does not exists. path: {}".format(avatar_video_path))

        return avatar_video_path

    def _get_mel_chunks(self, mel):
        mel_step_size = self.args.mel_step_size
        mel_idx_multiplier = 80. / self.args.fps

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1
        return mel_chunks

    def _base_frame_generator(self, mels):
        face_det_results = self._get_avatar_face_detection_results()
        frames = self._get_avatar_videos_frames()

        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        for i, m in enumerate(mels):
            idx = 0 if self.args.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.args.img_size, self.args.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.args.img_size // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def _lip_sync_generator(self, frame_generator):
        for i, (img_batch, mel_batch, frames, coords) in enumerate(frame_generator):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.lip_model.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.lip_model.device)

            # frame prediction using lip sync AI @predict
            pred = self.lip_model(mel_batch, img_batch).result()

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            predicted_frames = []
            for lip_frame, base_frame, face_coord in zip(pred, frames, coords):
                y1, y2, x1, x2 = face_coord
                p = cv2.resize(lip_frame.astype(np.uint8), (x2 - x1, y2 - y1))

                base_frame[y1:y2, x1:x2] = p
                predicted_frames.append(base_frame)
            yield np.array(predicted_frames)

    def _synced_frame_generator(self, audio_path):
        wav, sample_rate = librosa.core.load(audio_path, sr=16000)
        mel = audio.melspectrogram(wav)
        mel_chunks = self._get_mel_chunks(mel)
        frame_gen = NonBlockingLookaheadGenerator(self._base_frame_generator(mel_chunks))
        return NonBlockingLookaheadGenerator(self._lip_sync_generator(frame_gen))

    def get_idle_stream(self):
        return self._get_avatar_video()

    def _frame_buffer(self, audio_path):
        def frame_buffer_generator():
            frame_batch = []
            frame_generator = self._synced_frame_generator(audio_path)
            for _, prediction_batch in enumerate(frame_generator):
                for frame in prediction_batch:
                    frame_batch.append(frame)
                    if len(frame_batch) == self.args.buffer_size:
                        yield np.array(frame_batch)
                        frame_batch = []
                if frame_batch:
                    yield np.array(frame_batch)

        return frame_buffer_generator()

    def frame_buffer(self, audio_path, **kwargs):
        self.update_args(kwargs)
        return self._frame_buffer(audio_path)

    def video_buffer(self, audio_path, **kwargs):
        self.update_args(kwargs)
        audio_clip = AudioFileClip(audio_path)
        audio_time = self.args.buffer_size / self.args.fps
        for idx, fr24 in enumerate(self._frame_buffer(audio_path)):
            video_clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in fr24], fps=24)
            start_time = idx * audio_time
            end_time = (idx + 1) * audio_time
            video_clip = video_clip.with_audio(audio_clip.subclipped(start_time, min(end_time, audio_clip.duration)))
            yield video_clip

    def update_args(self, new_args):
        if new_args:
            self.args = SimpleNamespace(**{**vars(self.args), **new_args})

    def video_writer(self, audio_path, output):
        temp = f'_temp_avatar_{time.strftime("%Y%m%d_%H%M%S")}_{random.randint(0, 99999)}.avi'
        generator = self._synced_frame_generator(audio_path)
        first_frame_batch = next(generator)
        frame_h, frame_w = first_frame_batch[0].shape[:-1]
        out = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*'DIVX'), self.args.fps, (frame_w, frame_h))
        print(f"write file p1: for {self.id}")
        for frame in first_frame_batch:
            out.write(frame)
        for frame_batch in generator:
            print(f"write file p2: for {self.id}")
            for frame in frame_batch:
                out.write(frame)
        out.release()
        command = f'ffmpeg -y -i {audio_path} -i {temp} -strict -2 -q:v 1 {output}'
        subprocess.call(command, shell=platform.system() != 'Windows')
        os.remove(temp)


class AvatarManager:
    def __init__(self, avatar_model):
        self.avatar_cache = {}
        self.avatar_model = avatar_model

    def get_avatar(self, persona):
        if persona not in self.avatar_cache:
            avatar = Avatar(persona, self.avatar_model)
            avatar.init()
            self.avatar_cache[persona] = avatar
        else:
            avatar = self.avatar_cache[persona]
        return avatar
