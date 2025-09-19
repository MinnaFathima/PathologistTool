import torch
from model import HybridCNNViT
from utils import tile_image, pil_to_tensor, stitch_heatmap
from gradcam import GradCAM, vit_attention_map
from torchvision import transforms
from PIL import Image
import numpy as np

# transforms to match training
cnn_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

vit_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

class InferenceEngine:
    def __init__(self, model_checkpoint=None, device=None, tile_size=224, stride=224):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if model_checkpoint is None:
            raise ValueError("Provide model checkpoint path")
        self.model = HybridCNNViT().to(self.device)
        state = torch.load(model_checkpoint, map_location=self.device)
        if 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])
        else:
            self.model.load_state_dict(state)
        self.model.eval()
        self.tile_size = tile_size
        self.stride = stride

        # create GradCAM for cnn part: use layer4 of resnet
        self.gradcam = GradCAM(self.model.cnn, target_layer=self.model.cnn.layer4)

    def predict_whole_image(self, pil_img):
        tiles, coords = tile_image(pil_img, tile_size=self.tile_size, stride=self.stride)
        heatmaps = []
        tile_preds = []
        for tile in tiles:
            # prepare inputs
            cnn_t = pil_to_tensor(tile, cnn_transform).to(self.device)
            vit_t = pil_to_tensor(tile, vit_transform).to(self.device)
            # forward
            with torch.no_grad():
                out = self.model(cnn_t, vit_t)
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()
                pred = probs[1]
                tile_preds.append(pred)
            # gradcam
            self.model.cnn.zero_grad()
            cam = self.gradcam(cnn_t)
            # cam is (H, W) normalized
            heatmaps.append(cam)
        stitched = stitch_heatmap(coords, heatmaps, pil_img.size, tile_size=self.tile_size)
        avg_confidence = float(np.mean(tile_preds))
        return stitched, avg_confidence

    def get_vit_attention_map_for_tile(self, tile):
        vit_t = pil_to_tensor(tile, vit_transform).to(self.device)
        attn = vit_attention_map(self.model.vit, vit_t)
        return attn

    def close(self):
        self.gradcam.remove_hooks()