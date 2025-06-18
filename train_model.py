# train_model.py

import os
import torch # deep learning
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt # Plotting
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler #(Handle Class Imbalance)
from torchvision.models import resnet18, ResNet18_Weights  # PreTrained Models
from collections import Counter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset path
dataset_path = '/home/eaint/Downloads/archive (1)/raw-img'

# Transforms (For Training)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)), #(NN requires input image to 128*128 pixels(same size))
    transforms.RandomHorizontalFlip(), #(default probability is 0.5 (Data augmentation technique  left face and right face are the same class))
    transforms.RandomRotation(15),#Random rotates the images within +-15 Deg
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #more robust the different lighting condition and color variations
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]) #mean and std deviation for each RGB channels
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), #(H,C,W to C,H,W)(Height,Width,Color)
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset without transform yet
full_dataset = datasets.ImageFolder(dataset_path)
class_names = full_dataset.classes
print("Classes:", class_names)

# Split data
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

# Apply different transforms
train_data.dataset.transform = train_transform
val_data.dataset.transform = val_transform

# Optional: Handle class imbalance with weighted sampling
targets = [sample[1] for sample in full_dataset.samples]
class_counts = torch.bincount(torch.tensor(targets))
class_weights = 1.0 / class_counts.float()
train_targets = [train_data[i][1] for i in range(len(train_data))]
sample_weights = class_weights[torch.tensor(train_targets)]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Dataloaders
train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

def train_model():
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(30):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()

        print(f"Epoch [{epoch+1}/30], Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Plot
    epochs = range(1, 31)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='green', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Save model and class names
    torch.save(model.state_dict(), 'animal_cnn.pth')
    with open('class_names.txt', 'w') as f:
        for item in class_names:
            f.write(f"{item}\n")
    print("Model saved as 'animal_cnn.pth' and class names saved.")

if __name__ == '__main__':
    train_model()
