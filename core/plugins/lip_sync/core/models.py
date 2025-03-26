import torch
from onnx2torch import convert

from core.interfaces.base_ai import NonBlockingAIModel
from core.plugins.lip_sync.wave2lip.models import Wav2Lip
from manifest import Manifest


class AvatarWave2LipModel(NonBlockingAIModel):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.model_path = Manifest().get('avatar')['model_path']
        super().__init__()

    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint

    def load_model(self):
        if self.model_path.endswith('.onnx'):
            model = convert(self.model_path)
        else:
            model = Wav2Lip()
            checkpoint = self._load(self.model_path)
            s = checkpoint["state_dict"]
            new_s = {}
            for k, v in s.items():
                new_s[k.replace('module.', '')] = v
            model.load_state_dict(new_s)
        model = model.to(self.device)
        return model.eval()
