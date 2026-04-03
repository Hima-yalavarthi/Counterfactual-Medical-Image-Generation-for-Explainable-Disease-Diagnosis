import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.data_loader import get_dataloaders

def denormalize(img):
    """Denormalizes ImageNet normalized images."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def verify_and_visualize():
    data_dir = 'data/chest_xray'
    batch_size = 8
    
    print(f"Loading data from {data_dir}...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir, batch_size=batch_size)
    
    print(f"Classes: {classes}")
    
    # Get a batch of training data
    images, labels = next(iter(train_loader))
    
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Distribution: {torch.bincount(labels).tolist()}")
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for i in range(4):
        ax = axes[i]
        img = denormalize(images[i])
        ax.imshow(img)
        ax.set_title(classes[labels[i]])
        ax.axis('off')
    
    output_path = 'results/data_samples.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Sample images saved to {output_path}")

if __name__ == "__main__":
    verify_and_visualize()
