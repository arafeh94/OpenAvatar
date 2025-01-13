from onnx2torch import convert
import torch

from external.core.interfaces.base_ai import NonBlockingAIModel
from manifest import Manifest


class AvatarWave2LipModel(NonBlockingAIModel):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.model_path = Manifest().get('avatar')['gan_model_path']
        super().__init__()

    def load_model(self):
        onnx_model_path = self.model_path
        model = convert(onnx_model_path)
        model = model.to(self.device)
        return model.eval()
