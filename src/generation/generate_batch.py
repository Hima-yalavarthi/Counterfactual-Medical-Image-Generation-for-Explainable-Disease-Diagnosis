import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.cyclegan import ResNetGenerator

def generate_batch(predictions_csv, model_dir, output_dir, data_root='data/chest_xray', mode='P2N'):
    """
    Generates counterfactuals for the dataset.
    mode='P2N': Pneumonia -> Normal (using G_BA.pth)
    mode='N2P': Normal -> Pneumonia (using G_AB.pth)
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} | Mode: {mode}")

    # 1. Load Data
    df = pd.read_csv(predictions_csv)
    
    if mode == 'P2N':
        target_df = df[(df['predicted_label'] == 'PNEUMONIA') & (df['split'] == 'train')]
        model_path = os.path.join(model_dir, 'G_BA.pth')
        print(f"Generating Normal counterfactuals for {len(target_df)} Pneumonia images...")
    else:
        target_df = df[(df['predicted_label'] == 'NORMAL') & (df['split'] == 'train')]
        model_path = os.path.join(model_dir, 'G_AB.pth')
        print(f"Generating Pneumonia counterfactuals for {len(target_df)} Normal images...")

    # 2. Load Model
    model = ResNetGenerator(n_blocks=6).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded: {model_path}")
    else:
        print(f"Error: Model NOT found at {model_path}")
        return
    model.eval()

    # 3. Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    inv_normalize = transforms.Normalize(
        mean=[-1, -1, -1],
        std=[2, 2, 2]
    )

    os.makedirs(output_dir, exist_ok=True)

    # 4. Generate
    with torch.no_grad():
        for _, row in tqdm(target_df.iterrows(), total=len(target_df)):
            img_path = os.path.join(data_root, row['filename'])
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            fake_img = model(img_tensor)
            
            fake_img = inv_normalize(fake_img.squeeze(0)).cpu()
            fake_pill = transforms.ToPILImage()(fake_img)
            
            out_path = os.path.join(output_dir, row['filename'])
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fake_pill.save(out_path)

    print(f"Counterfactuals saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_csv', type=str, default='results/predictions.csv')
    parser.add_argument('--model_dir', type=str, default='src/models/cyclegan')
    parser.add_argument('--output_dir', type=str, default='results/counterfactuals')
    parser.add_argument('--mode', type=str, default='P2N', choices=['P2N', 'N2P'])
    args = parser.parse_args()
    
    generate_batch(args.predictions_csv, args.model_dir, args.output_dir, mode=args.mode)
