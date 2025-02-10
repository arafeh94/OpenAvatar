import logging
import os
import pickle
from types import SimpleNamespace

import numpy as np
import torch

from external.core.utils.text_split import TextSampler
from external.plugins.lip_sync.wave2lip.audio import melspectrogram
from external.plugins.text2speech import Text2Speech, Audio
from manifest import Manifest
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

    def lip_synced_frame_generator(self, audio: Audio):
        mel = melspectrogram(audio.samples)
        mel_chunks = self._get_mel_chunks(mel)
        frame_gen = NonBlockingLookaheadGenerator(self._base_frame_generator(mel_chunks))
        return NonBlockingLookaheadGenerator(self._lip_sync_generator(frame_gen))

    def get_idle_stream(self):
        return self._get_avatar_video()

    def _frame_buffer(self, audio: Audio):
        def frame_buffer_generator():
            frame_batch = []
            frame_generator = self.lip_synced_frame_generator(audio)
            for _, prediction_batch in enumerate(frame_generator):
                for frame in prediction_batch:
                    frame_batch.append(frame)
                    if len(frame_batch) == self.args.buffer_size:
                        yield np.array(frame_batch)
                        frame_batch = []
                if frame_batch:
                    yield np.array(frame_batch)

        return frame_buffer_generator()

    def frame_buffer(self, audio: Audio, **kwargs):
        self.update_args(kwargs)
        return NonBlockingLookaheadGenerator(self._frame_buffer(audio))

    def stream(self, audio: Audio, **kwargs):
        self.update_args(kwargs)
        return AvatarBuffer(NonBlockingLookaheadGenerator(self._frame_buffer(audio)))

    def tts_buffer(self, tts, text, **kwargs):
        return AvatarTTS(self, tts).buffer(text, **kwargs)

    def update_args(self, new_args):
        if new_args:
            self.args = SimpleNamespace(**{**vars(self.args), **new_args})


class AvatarTTS:
    def __init__(self, avatar: Avatar, tts_convertor: Text2Speech):
        self.tts_convertor = tts_convertor
        self.avatar = avatar

    def tts(self, text, **kwargs):
        max_batch_size = kwargs.get('max_text_sample', Manifest().query('tts.max_text_sample', 200))
        text_sampler = TextSampler(text, max_batch_size)
        while True:
            speech_text = next(text_sampler)
            if speech_text is None:
                break
            audio = self.tts_convertor.convert(speech_text, **kwargs).as_audio()
            yield self.avatar.stream(audio, **kwargs), audio

    def buffer(self, text, **kwargs):
        return NonBlockingLookaheadGenerator(self.tts(text, **kwargs))


class AvatarManager:
    def __init__(self, avatar_model, tts_convertor: Text2Speech):
        self.avatar_cache = {}
        self.avatar_model = avatar_model
        self.tts_convertor = tts_convertor

    def get_avatar(self, persona):
        if persona not in self.avatar_cache:
            avatar = Avatar(persona, self.avatar_model)
            avatar.init()
            self.avatar_cache[persona] = avatar
        else:
            avatar = self.avatar_cache[persona]
        return avatar

    def tts_buffer(self, persona, text, **kwargs):
        return self.get_avatar(persona).tts_buffer(self.tts_convertor, text, **kwargs)


class AvatarBuffer:
    def __init__(self, frame_buffer):
        self._frame_buffer = frame_buffer
        self._frame_stream = np.array([])
        self._frame_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if not self._frame_stream.any() or self._frame_index >= len(self._frame_stream):
                self._frame_index = 0
                self._frame_stream = next(self._frame_buffer)
        except StopIteration:
            raise StopIteration

        self._frame_index += 1
        return self._frame_stream[self._frame_index - 1]
