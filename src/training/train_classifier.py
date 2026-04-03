import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import copy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.data_loader import get_dataloaders
from models.classifier import get_resnet18_classifier

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device='cuda'):
    since = time.time()
    
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # TensorBoard disabled due to system compatibility issues
    # writer = SummaryWriter('results/logs/classifier')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            all_preds = []
            all_labels = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'val':
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.data.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            # writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # Deep copy the model if it has better validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Calculate detailed metrics for validation
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='binary'
                )
                print(f'New Best Val Acc! Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    # writer.close()
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Pneumonia Classifier')
    parser.add_argument('--data_dir', type=str, default='data/chest_xray', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='src/models/best_classifier.pth', help='Path to save best model')
    args = parser.parse_args()

    # Setup parameters
    data_dir = args.data_dir
    model_save_path = args.save_path
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Get Dataloaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # 2. Initialize Model
    model = get_resnet18_classifier(num_classes=len(classes), pretrained=True).to(device)

    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Train
    print("Starting training...")
    best_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, device=device)

    # 5. Save best model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    # 6. Final Evaluation on Test Set
    print("\nFinal Evaluation on Test Set:")
    best_model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = best_model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())

    acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()
