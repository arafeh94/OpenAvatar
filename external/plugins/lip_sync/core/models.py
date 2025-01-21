import numpy as np
import torch
from onnx2torch import convert

from external.core.interfaces.base_ai import NonBlockingAIModel, AIModel
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


# class UpscalerMode(AIModel):
#     def __init__(self):
#         self.model_path = Manifest().get('avatar')['esrgan_model_path']
#         super().__init__()
#
#     def load_model(self):
#         model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
#         model.load_state_dict(torch.load(self.model_path)['params'], strict=True)
#         model.eval()
#         model = model.cuda()
#         return model
#
#     def upscale(self, image):
#         img = image.astype(np.float32) / 255.
#         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#         img = img.unsqueeze(0).cuda()
#         with torch.no_grad():
#             output = self(img)
#         output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#         output = (output * 255.0).round().astype(np.uint8)
#         return output
