import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
import pandas as pd
from tqdm import tqdm
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.classifier import get_resnet18_classifier

def validate_flip_rate(predictions_csv, cf_dir, model_path):
    """
    Calculates the 'Flip Rate': % of counterfactuals that successfully 
    changed the classifier's prediction to the target class.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    df = pd.read_csv(predictions_csv)
    # Only evaluate training set counterfactuals for now (matching generate_batch)
    df = df[df['split'] == 'train']
    
    # 2. Load Model
    model = get_resnet18_classifier(pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Classifier model loaded.")
    else:
        print(f"Error: Model NOT found at {model_path}")
        return
    model = model.to(device)
    model.eval()

    # 3. Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Classifier input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results = []
    success_count = 0
    total_count = 0

    print("Validating flip rate...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            cf_path = os.path.join(cf_dir, row['filename'])
            if not os.path.exists(cf_path):
                continue
                
            total_count += 1
            
            # Load and predict on counterfactual
            img = Image.open(cf_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            prob = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(prob)
            
            # Label mapping: 0: NORMAL, 1: PNEUMONIA
            target_class = 0 if row['predicted_label'] == 'PNEUMONIA' else 1
            
            is_success = (pred_class == target_class)
            if is_success:
                success_count += 1
                
            results.append({
                'filename': row['filename'],
                'original_pred': row['predicted_label'],
                'new_pred': 'NORMAL' if pred_class == 0 else 'PNEUMONIA',
                'success': bool(is_success),
                'prob_norm': float(prob[0]),
                'prob_pneu': float(prob[1])
            })

    flip_rate = (success_count / total_count) if total_count > 0 else 0
    print(f"\nFinal Flip Rate: {flip_rate:.2%} ({success_count}/{total_count})")

    # Update evaluation_metrics.json
    metrics_path = 'results/evaluation_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {}
        
    metrics_data['flip_rate_summary'] = {
        'total_evaluated': total_count,
        'success_count': success_count,
        'flip_rate': flip_rate
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"Metrics updated in {metrics_path}")

if __name__ == "__main__":
    validate_flip_rate(
        predictions_csv='results/predictions.csv',
        cf_dir='results/counterfactuals',
        model_path='src/models/best_classifier.pth'
    )
