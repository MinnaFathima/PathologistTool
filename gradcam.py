import torch
import torch.nn.functional as F
import numpy as np

# Grad-CAM for ResNet part (CNN branch)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        handle_f = self.target_layer.register_forward_hook(forward_hook)
        handle_b = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([handle_f, handle_b])

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        # input_tensor: (1, C, H, W)
        outputs = self.model(input_tensor)
        if class_idx is None:
            class_idx = outputs.argmax().item()
        loss = outputs[0, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        grads = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        weights = grads.mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        cam_np = cam.cpu().numpy()
        return cam_np

# Simple attention rollout for ViT to produce a CAM-like map
import numpy as np

def vit_attention_map(vit_model, vit_inputs):
    # vit_model is transformers.ViTModel; vit_inputs shape (1, C, H, W)
    # We will grab the attentions via forward pass with output_attentions=True
    outputs = vit_model(vit_inputs, output_attentions=True)
    attentions = outputs.attentions  # tuple of (layers) each (batch, heads, tokens, tokens)

    # average heads
    attn_mat = torch.stack([a.mean(dim=1) for a in attentions], dim=0)  # (layers, batch, tokens, tokens) averaged over heads
    attn_mat = attn_mat[0]  # (batch, tokens, tokens)? We'll do iterative product
    # iterative multiplication for rollout
    result = torch.eye(attn_mat.size(-1)).to(attn_mat.device)
    for layer_attn in attn_mat:  # layer_attn: (tokens, tokens)
        layer_attn = layer_attn + torch.eye(layer_attn.size(-1)).to(layer_attn.device)
        layer_attn = layer_attn / layer_attn.sum(dim=-1, keepdim=True)
        result = layer_attn @ result
    # result now has influence of each token on CLS token etc.
    # We want the attention of CLS token to patches
    # CLS token is token 0; patches begin at 1
    cls_attn = result[:, 0]
    # drop CLS token and reshape patch tokens to 2D
    cls_attn = cls_attn[1:]
    return cls_attn.cpu().numpy()