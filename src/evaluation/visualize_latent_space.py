import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.classifier import get_resnet18_classifier

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    filenames = []
    
    # Hook to get features from avgpool
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model.avgpool.register_forward_hook(get_activation('avgpool'))
    
    with torch.no_grad():
        for img_batch, label_batch, fnames in tqdm(dataloader):
            img_batch = img_batch.to(device)
            _ = model(img_batch)
            feat = activation['avgpool'].squeeze().cpu().numpy()
            if len(feat.shape) == 1: # Handle batch size 1
                feat = feat.reshape(1, -1)
            features.append(feat)
            labels.extend(label_batch)
            filenames.extend(fnames)
            
    return np.concatenate(features), labels, filenames

def run_latent_visualization(predictions_csv, cf_dir, model_path, data_root='data/chest_xray'):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data List
    df = pd.read_csv(predictions_csv)
    df = df[df['split'] == 'train'].sample(n=min(500, len(df)), random_state=42) # Sample for speed
    
    # 2. Load Model
    classifier = get_resnet18_classifier(pretrained=False)
    if os.path.exists(model_path):
        classifier.load_state_dict(torch.load(model_path, map_location=device))
        print("Classifier loaded.")
    classifier = classifier.to(device)
    classifier.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_features = []
    all_labels = []
    all_types = []
    all_filenames = []

    print("Extracting features...")
    with torch.no_grad():
        # Hook for avgpool
        feature_list = []
        def hook_fn(module, input, output):
            feature_list.append(output.squeeze().cpu().numpy())
        handle = classifier.avgpool.register_forward_hook(hook_fn)

        for _, row in tqdm(df.iterrows(), total=len(df)):
            orig_path = os.path.join(data_root, row['filename'])
            cf_path = os.path.join(cf_dir, row['filename'])
            
            if not os.path.exists(orig_path): continue
            
            # Original Image
            img = transform(Image.open(orig_path).convert('RGB')).unsqueeze(0).to(device)
            feature_list.clear()
            _ = classifier(img)
            all_features.append(feature_list[0])
            all_labels.append(row['true_label'])
            all_types.append('Original')
            all_filenames.append(row['filename'])
            
            # Counterfactual Image
            if os.path.exists(cf_path):
                img_cf = transform(Image.open(cf_path).convert('RGB')).unsqueeze(0).to(device)
                feature_list.clear()
                _ = classifier(img_cf)
                all_features.append(feature_list[0])
                all_labels.append('NORMAL' if row['predicted_label'] == 'PNEUMONIA' else 'PNEUMONIA')
                all_types.append('Counterfactual')
                all_filenames.append(f"cf_{row['filename']}")

        handle.remove()

    features_np = np.array(all_features)
    print(f"Features extracted: {features_np.shape}")

    # 3. Run PCA
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings = pca.fit_transform(features_np)

    # 4. Save results
    res_df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'label': all_labels,
        'type': all_types,
        'filename': all_filenames
    })
    
    os.makedirs('outputs', exist_ok=True)
    res_df.to_csv('results/latent_coordinates.csv', index=False)
    print("Latent coordinates saved to results/latent_coordinates.csv")

    # 5. Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=res_df, x='x', y='y', hue='label', style='type', alpha=0.6)
    plt.title("PCA Visualization of Classifier Latent Space")
    plt.savefig('results/latent_space_pca.png')
    print("Static plot saved to results/latent_space_pca.png")

if __name__ == "__main__":
    run_latent_visualization(
        predictions_csv='results/predictions.csv',
        cf_dir='results/counterfactuals',
        model_path='src/models/best_classifier.pth'
    )
