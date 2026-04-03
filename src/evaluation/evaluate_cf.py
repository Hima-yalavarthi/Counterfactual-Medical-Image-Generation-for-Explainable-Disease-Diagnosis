import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import lpips
import json

def load_image(path, transform):
    image = Image.open(path).convert('RGB')
    return transform(image).unsqueeze(0)

def evaluate_counterfactuals(predictions_csv, cf_dir, data_root='data/chest_xray'):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    df = pd.read_csv(predictions_csv)
    # Filter for cases that have a counterfactual generated
    pneumonia_df = df[(df['predicted_label'] == 'PNEUMONIA') & (df['split'] == 'train')]
    
    # 2. Setup Metrics
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    results = []
    ssim_scores = []
    lpips_scores = []

    print(f"Evaluating {len(pneumonia_df)} counterfactuals...")
    
    for _, row in tqdm(pneumonia_df.iterrows(), total=len(pneumonia_df)):
        orig_path = os.path.join(data_root, row['filename'])
        cf_path = os.path.join(cf_dir, row['filename'])
        
        if not os.path.exists(orig_path) or not os.path.exists(cf_path):
            continue
            
        # Load images
        orig_img = Image.open(orig_path).convert('RGB').resize((128, 128))
        cf_img = Image.open(cf_path).convert('RGB').resize((128, 128))
        
        # SSIM (Structural Similarity)
        # Convert to grayscale for SSIM
        orig_gray = np.array(orig_img.convert('L'))
        cf_gray = np.array(cf_img.convert('L'))
        score_ssim = ssim(orig_gray, cf_gray)
        ssim_scores.append(score_ssim)
        
        # LPIPS (Perceptual Similarity)
        orig_tensor = transform(orig_img).to(device)
        cf_tensor = transform(cf_img).to(device)
        
        with torch.no_grad():
            score_lpips = loss_fn_vgg(orig_tensor, cf_tensor).item()
        lpips_scores.append(score_lpips)
        
        results.append({
            'filename': row['filename'],
            'ssim': score_ssim,
            'lpips': score_lpips
        })

    # Summary Statistics
    summary = {
        'total_evaluated': len(results),
        'mean_ssim': np.mean(ssim_scores) if ssim_scores else 0,
        'std_ssim': np.std(ssim_scores) if ssim_scores else 0,
        'mean_lpips': np.mean(lpips_scores) if lpips_scores else 0,
        'std_lpips': np.std(lpips_scores) if lpips_scores else 0
    }

    print("\nEvaluation Summary:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Save results
    output_path = 'results/evaluation_metrics.json'
    with open(output_path, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=4)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    evaluate_counterfactuals(
        predictions_csv='results/predictions.csv',
        cf_dir='results/counterfactuals'
    )
