import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations by these pooled gradients
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # ReLU on the heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap
        heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1.0
        
        return heatmap.detach().cpu().numpy()

    @staticmethod
    def overlay_heatmap(heatmap, original_img, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Blend with original image
        overlayed_img = heatmap * alpha + original_img * (1 - alpha)
        overlayed_img = np.uint8(overlayed_img)
        
        return overlayed_img
