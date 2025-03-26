import torch
from transformers import pipeline
from datasets import load_dataset
from core.interfaces.base_tts import Text2Speech
from core.interfaces.va import VoiceConvertable
from manifest import Manifest


class MicrosoftText2Speech(Text2Speech):
    def __init__(self, device='cuda'):
        batch_size = Manifest().query('tts.prediction_batch_size', 8000)
        self.model = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device, batch_size=batch_size)
        self.dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    def convert(self, text, **kwargs) -> VoiceConvertable:
        if 'voice_id' not in kwargs:
            raise Exception("MicrosoftText2Speech requires voice_id parameter")
        voice_id = kwargs['voice_id']
        speaker_embeddings = torch.tensor(self.dataset[voice_id]["xvector"]).unsqueeze(0)
        voice = self.model(text, forward_params={"speaker_embeddings": speaker_embeddings})
        return VoiceConvertable(voice)
