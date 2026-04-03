import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.classifier import get_resnet18_classifier
from utils.gradcam import GradCAM

def generate_gradcam_batch(predictions_csv, model_path, output_dir, data_root='data/chest_xray'):
    """
    Generates Grad-CAM heatmaps for all Pneumonia images in the dataset.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    df = pd.read_csv(predictions_csv)
    # Filter for Pneumonia cases (predicted or true, but here we use predicted to explain AI focus)
    pneumonia_df = df[df['predicted_label'] == 'PNEUMONIA']
    print(f"Generating Grad-CAM heatmaps for {len(pneumonia_df)} images...")

    # 2. Load Model
    model = get_resnet18_classifier(pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Classifier model loaded.")
    else:
        print(f"Warning: Model not found at {model_path}. Using initial weights.")
    
    model = model.to(device)
    model.eval()

    # 3. Setup Grad-CAM (Targeting layer4 for ResNet-18)
    cam = GradCAM(model, model.layer4)

    # 4. Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Standardize size for heatmap generation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    os.makedirs(output_dir, exist_ok=True)

    # 5. Generate
    for _, row in tqdm(pneumonia_df.iterrows(), total=len(pneumonia_df)):
        img_path = os.path.join(data_root, row['filename'])
        if not os.path.exists(img_path):
            continue
            
        # Load and transform image
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Generate heatmap (class 1 is Pneumonia)
        heatmap = cam.generate_heatmap(img_tensor, target_class=1)
        
        # Overlay heatmap
        original_img_np = np.array(img_pil.resize((256, 256)))
        heatmap_overlay = cam.overlay_heatmap(heatmap, original_img_np)
        
        # Save to output_dir
        out_path = os.path.join(output_dir, row['filename'])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))

    print(f"Grad-CAM heatmaps saved to {output_dir}")

if __name__ == "__main__":
    generate_gradcam_batch(
        predictions_csv='results/predictions.csv',
        model_path='src/models/best_classifier.pth',
        output_dir='results/gradcam'
    )
