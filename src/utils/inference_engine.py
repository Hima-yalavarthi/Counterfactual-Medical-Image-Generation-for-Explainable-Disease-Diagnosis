import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from src.utils.gradcam import GradCAM

def run_live_inference(image, model, device='cpu'):
    """
    Performs real-time inference, Grad-CAM generation, and stability analysis.
    """
    model.eval()
    
    # 1. Pre-process
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 2. Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
        confidence = np.max(probs)
        predicted_idx = np.argmax(probs)
        predicted_label = 'PNEUMONIA' if predicted_idx == 1 else 'NORMAL'
        
    # 3. Grad-CAM
    # Target the last layer of ResNet-18
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate_heatmap(input_tensor, target_class=predicted_idx)
    
    # Prepare original image for overlay
    orig_np = np.array(image.convert('RGB'))
    overlay = GradCAM.overlay_heatmap(heatmap, orig_np)
    
    # 4. Stability Analysis (Mini-version for real-time)
    stability_score = 0.0
    num_iters = 5
    noise_lvl = 0.05
    
    with torch.no_grad():
        for _ in range(num_iters):
            noise = torch.randn_like(input_tensor) * noise_lvl
            noisy_input = input_tensor + noise
            noisy_output = model(noisy_input)
            noisy_probs = F.softmax(noisy_output, dim=1).squeeze().cpu().numpy()
            # If the prediction remains the same, it's stable
            if np.argmax(noisy_probs) == predicted_idx:
                stability_score += 1.0
                
    stablity = stability_score / num_iters
    
    return {
        'label': predicted_label,
        'confidence': confidence,
        'probs': {
            'NORMAL': float(probs[0]),
            'PNEUMONIA': float(probs[1])
        },
        'heatmap_img': Image.fromarray(overlay),
        'stability_score': stablity
    }
