import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

def get_data_transforms():
    """Returns training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_weights_for_balanced_classes(dataset):
    """Computes weights for each sample to handle class imbalance."""
    counts = [0] * len(dataset.classes)
    for _, target in dataset.samples:
        counts[target] += 1
    
    counts = torch.tensor(counts).float()
    weights = 1.0 / counts
    sample_weights = [weights[target] for _, target in dataset.samples]
    return sample_weights

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Creates DataLoaders for train, val, and test sets.
    Uses WeightedRandomSampler for training to balance classes.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    
    # Handle Class Imbalance in Training Set
    sample_weights = get_weights_for_balanced_classes(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes
