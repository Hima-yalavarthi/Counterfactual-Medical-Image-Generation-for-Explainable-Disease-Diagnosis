import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.classifier import get_resnet18_classifier

def get_inference_transforms():
    """Returns standard ImageNet normalization transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def batch_inference(data_dir, model_path, output_csv):
    """
    Runs inference on all images in data_dir and saves to output_csv.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    # We need to know num_classes (2 for this dataset)
    model = get_resnet18_classifier(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.")

    # 2. Get All Images
    transform = get_inference_transforms()
    
    # We'll traverse train, val, test
    results = []
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        dataset = datasets.ImageFolder(split_dir, transform=transform)
        # Using a custom DataLoader to get filenames
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        classes = dataset.classes
        print(f"Processing {split} split ({len(dataset)} images)...")
        
        # To get filenames, we need to access dataset.samples
        samples = dataset.samples
        sample_idx = 0
        
        with torch.no_grad():
            for inputs, _ in tqdm(loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, 1)
                
                # Get filenames and other info
                for i in range(len(inputs)):
                    img_path, true_label_idx = samples[sample_idx]
                    results.append({
                        'filename': os.path.relpath(img_path, data_dir),
                        'split': split,
                        'true_label': classes[true_label_idx],
                        'predicted_label': classes[preds[i].item()],
                        'confidence': conf[i].item(),
                        'correct': (true_label_idx == preds[i].item()),
                        'prob_normal': probs[i][0].item(),
                        'prob_pneumonia': probs[i][1].item()
                    })
                    sample_idx += 1

    # 3. Save Results
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    print(df.head())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/chest_xray')
    parser.add_argument('--model_path', type=str, default='src/models/best_classifier.pth')
    parser.add_argument('--output_csv', type=str, default='results/predictions.csv')
    args = parser.parse_args()
    
    batch_inference(args.data_dir, args.model_path, args.output_csv)
