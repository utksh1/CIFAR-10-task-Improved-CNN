import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(subset_size=10000, batch_size=64, use_augmentation=False):
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ] + base_transforms)
    else:
        train_transform = transforms.Compose(base_transforms)

    test_transform = transforms.Compose(base_transforms)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    indices = torch.randperm(len(trainset))[:subset_size]
    train_subset = Subset(trainset, indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class SimpleCNN(nn.Module):
    def __init__(self, use_batchnorm=False):
        super(SimpleCNN, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

def train_model(model, train_loader, test_loader, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        history['loss'].append(running_loss / len(train_loader))
        history['acc'].append(acc)
        print(f"Epoch {epoch+1}: Loss {history['loss'][-1]:.4f}, Acc {acc:.2f}%")
        
    return history

def plot_results(baseline_hist, improved_hist):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(baseline_hist['loss'], label='Baseline')
    plt.plot(improved_hist['loss'], label='Improved')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(baseline_hist['acc'], label='Baseline')
    plt.plot(improved_hist['acc'], label='Improved')
    plt.title('Test Accuracy (%)')
    plt.legend()
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/comparison_plot.png')
    plt.show()

if __name__ == "__main__":
    print("Running baseline task...")
    train_loader, test_loader = get_dataloaders(use_augmentation=False)
    baseline_model = SimpleCNN(use_batchnorm=False)
    baseline_hist = train_model(baseline_model, train_loader, test_loader)
    
    print("\nRunning improved task...")
    train_loader_aug, _ = get_dataloaders(use_augmentation=True)
    improved_model = SimpleCNN(use_batchnorm=True)
    improved_hist = train_model(improved_model, train_loader_aug, test_loader)
    
    plot_results(baseline_hist, improved_hist)
