from onnx2torch import convert
import torch

from external.core.interfaces.base_ai import NonBlockingAIModel


class AvatarWave2LipModel(NonBlockingAIModel):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        super().__init__()

    def load_model(self):
        onnx_model_path = "/home/arafeh/PycharmProjects/avatar_rag_back_end/avatar/checkpoints/wav2lip_gan_96.onnx"
        model = convert(onnx_model_path)
        model = model.to(self.device)
        return model.eval()
