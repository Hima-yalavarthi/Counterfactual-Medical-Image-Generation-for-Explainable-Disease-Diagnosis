import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image

def generate_difference_maps(predictions_csv, cf_dir, data_root, output_dir):
    """
    Generates difference maps between original and counterfactual images.
    Highlights what regions were changed by the GAN.
    """
    # 1. Load Data
    df = pd.read_csv(predictions_csv)
    # Filter for cases with both original and counterfactual
    pneumonia_df = df[df['predicted_label'] == 'PNEUMONIA']
    print(f"Generating difference maps for {len(pneumonia_df)} images...")

    os.makedirs(output_dir, exist_ok=True)

    for _, row in tqdm(pneumonia_df.iterrows(), total=len(pneumonia_df)):
        orig_path = os.path.join(data_root, row['filename'])
        cf_path = os.path.join(cf_dir, row['filename'])
        
        if not os.path.exists(orig_path) or not os.path.exists(cf_path):
            continue
            
        # Load images
        orig_img = np.array(Image.open(orig_path).convert('L').resize((256, 256)))
        cf_img = np.array(Image.open(cf_path).convert('L').resize((256, 256)))
        
        # Calculate Absolute Difference
        diff = np.abs(orig_img.astype(np.float32) - cf_img.astype(np.float32))
        
        # Normalize for visualization
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        diff = np.uint8(255 * diff)
        
        # Apply Colormap (Jet)
        diff_heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        # (Optional) Overlay on original image for better context
        # Convert grayscale orig to BGR
        orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(orig_bgr, 0.7, diff_heatmap, 0.3, 0)
        
        # Save result
        out_path = os.path.join(output_dir, row['filename'])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, overlay)

    print(f"Difference maps saved to {output_dir}")

if __name__ == "__main__":
    generate_difference_maps(
        predictions_csv='results/predictions.csv',
        cf_dir='results/counterfactuals',
        data_root='data/chest_xray',
        output_dir='results/diff_maps'
    )
