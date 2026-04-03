import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_classifier(num_classes=2, pretrained=True):
    """
    Creates a ResNet18 model modified for binary classification.
    """
    # Load pretrained ResNet18
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18()
    
    # Replace the final fully connected layer
    # ResNet18's last layer is named 'fc'
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = get_resnet18_classifier()
    print(model)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")
