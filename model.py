import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from transformers import ViTModel

class HybridCNNViT(nn.Module):
    def __init__(self, pretrained_cnn=True, pretrained_vit=True):
        super(HybridCNNViT, self).__init__()
        # CNN backbone (ResNet18 with pretrained weights)
        if pretrained_cnn:
            self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()  # output 512-d

        # ViT backbone (pretrained from Hugging Face)
        if pretrained_vit:
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        else:
            self.vit = ViTModel.from_config(ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").config)

        # Combined fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, cnn_x, vit_x):
        cnn_feat = self.cnn(cnn_x)                 # (batch, 512)
        vit_outputs = self.vit(vit_x)
        # Some ViT versions use pooler_output, others use last_hidden_state
        vit_feat = getattr(vit_outputs, "pooler_output", None)
        if vit_feat is None:
            # fallback: mean of CLS token / last hidden state
            vit_feat = vit_outputs.last_hidden_state[:, 0, :]
        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        return self.fc(combined)

# helper to load weights
def load_model(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCNNViT().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # if the checkpoint saved module state dict
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model