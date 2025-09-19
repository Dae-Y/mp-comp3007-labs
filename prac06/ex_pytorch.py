"""
Machine Perception Prac06 - Machine Learning Part 2 (PyTorch Implementation)
Author: Daehwan Yeo

In this practical, we perform image classification on the CIFAR-10 dataset using deep learning 
approaches implemented in PyTorch. 

Dataset:
- CIFAR-10 contains 60,000 images (32x32 RGB), across 10 classes.
- Training: 50,000 images (we split into 40,000 train + 10,000 validation)
- Testing: 10,000 images

Exercises:
1. Build a basic CNN with Conv/Pooling layers and compare RMSprop vs Adam.
2. Add data augmentation and dropout, observe performance improvements.
3. Use transfer learning with VGG16 (and optionally ResNet50) as feature extractors.
4. Build a deeper CNN with batch normalization + dropout to maximize performance.

Outputs:
- Save confusion matrices and logs into `ex_pytorch/`.
- Track hyperparameters, metrics, and results in wandb dashboard.
"""

# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import models, transforms

# Ensure results folder exists
os.makedirs("ex_pytorch", exist_ok=True)


# -----------------------
# Exercise 1: Basic CNN
# -----------------------
def exercise1_pytorch_basic():
    """
    Trains and evaluates a simple CNN on CIFAR-10 using
    RMSprop and Adam optimizers, and saves confusion matrices.
    """
    print("\n--- Running Exercise 1: Basic CNN with W&B logging ---")

    # Start a new wandb run
    wandb.init(
        project="comp3007-prac06",
        entity="dae-y-dev-curtin-university",
        config={
            "batch_size": 64,
            "epochs": 5,
            "dataset": "CIFAR-10",
            "model": "BasicCNN",
        }
    )

    # Data
    transform = transforms.ToTensor()
    train_dataset_full = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_idx, val_idx = train_test_split(
        range(len(train_dataset_full)), test_size=0.2,
        stratify=train_dataset_full.targets, random_state=42
    )
    train_subset = torch.utils.data.Subset(train_dataset_full, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset_full, val_idx)

    batch_size = wandb.config["batch_size"]
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    class CNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * 4 * 4, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            return self.fc1(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_and_eval(opt_name, lr=0.001, epochs=5):
        model = CNNModel().to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=lr) if opt_name == "RMSprop" else optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for ep in range(epochs):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    _, pred = torch.max(out, 1)
                    total += y.size(0)
                    correct += (pred == y).sum().item()

            val_acc = 100 * correct / total
            wandb.log({f"{opt_name}_loss": total_loss/len(train_loader),
                       f"{opt_name}_val_acc": val_acc})
            print(f"{opt_name} Epoch {ep+1}, Loss {total_loss/len(train_loader):.4f}, Val Acc {val_acc:.2f}%")

        # test set
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                _, pred = torch.max(out, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        wandb.log({f"{opt_name}_test_acc": 100 * acc})
        return cm

    cm_rms = train_and_eval("RMSprop")
    cm_adam = train_and_eval("Adam")

    # Plot confusion matrices
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_rms, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - RMSprop")
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_adam, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Adam")
    plt.savefig("ex_pytorch/ex1_confusion_matrix.png")
    plt.close()

    wandb.finish()


# -------------------------------
# Exercise 2: Data Augmentation
# -------------------------------
def exercise2_pytorch_augmentation():
    """
    CNN with random horizontal flip and dropout.
    """
    print("\n--- Running Exercise 2: Data Augmentation ---")

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_tf = transforms.ToTensor()

    train_dataset_full = CIFAR10(root='./data', train=True, download=True, transform=train_tf)
    val_dataset = CIFAR10(root='./data', train=True, download=True, transform=test_tf)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_tf)

    train_idx, val_idx = train_test_split(range(len(train_dataset_full)), test_size=0.2,
                                          stratify=train_dataset_full.targets, random_state=42)
    train_subset = torch.utils.data.Subset(train_dataset_full, train_idx)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    class CNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Linear(128 * 4 * 4, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.dropout(x.view(-1, 128 * 4 * 4))
            return self.fc1(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for ep in range(10):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Test
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Augmentation + Dropout")
    plt.savefig("ex_pytorch/ex2_confusion_matrix.png")
    plt.close()


# --------------------------------
# Exercise 3: Pre-trained VGG16
# --------------------------------
def exercise3_pytorch_pretrained():
    """
    Transfer learning using VGG16 on CIFAR-10.
    (No saving of large model weights, just report validation and test results.)
    """
    print("\n--- Running Exercise 3: Pre-trained VGG16 ---")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset_full = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_idx, val_idx = train_test_split(range(len(train_dataset_full)), test_size=0.2,
                                          stratify=train_dataset_full.targets, random_state=42)
    train_subset = torch.utils.data.Subset(train_dataset_full, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset_full, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # pretrained model is so slow with CPU, let's skip
    vgg16 = models.vgg16(weights=None)  
    # vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    for p in vgg16.features.parameters():
        p.requires_grad = False
    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg16.to(device)

    optimizer = optim.Adam(vgg16.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for ep in range(3):
        vgg16.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = vgg16(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Evaluate
    correct, total = 0, 0
    vgg16.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = vgg16(X)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    acc = correct / total
    with open("ex_pytorch/ex3_val_loss.txt", "w") as f:
        f.write(f"Final Test Accuracy (VGG16): {acc:.4f}\n")
    print(f"Final Test Accuracy (VGG16): {acc:.4f}")


# -------------------------------------
# Exercise 4: Large CNN
# -------------------------------------
def exercise4_pytorch_large():
    """
    A larger CNN with batch norm and dropout on CIFAR-10.
    """
    print("\n--- Running Exercise 4: Large CNN ---")

    transform = transforms.ToTensor()
    train_dataset_full = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_idx, val_idx = train_test_split(range(len(train_dataset_full)), test_size=0.2,
                                          stratify=train_dataset_full.targets, random_state=42)
    train_subset = torch.utils.data.Subset(train_dataset_full, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset_full, val_idx)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    class LargeCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.drop1 = nn.Dropout(0.25)

            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.drop2 = nn.Dropout(0.25)

            self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn5 = nn.BatchNorm2d(128)
            self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
            self.bn6 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.drop3 = nn.Dropout(0.25)

            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.drop4 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.drop1(F.relu(self.bn2(self.conv2(x))))
            x = self.pool2(F.relu(self.bn3(self.conv3(x))))
            x = self.drop2(F.relu(self.bn4(self.conv4(x))))
            x = self.pool3(F.relu(self.bn5(self.conv5(x))))
            x = self.drop3(F.relu(self.bn6(self.conv6(x))))
            x = x.view(x.size(0), -1)
            x = self.drop4(F.relu(self.fc1(x)))
            return self.fc2(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LargeCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    for ep in range(10):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Evaluate
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Large CNN")
    plt.savefig("ex_pytorch/ex4_confusion_matrix.png")
    plt.close()


# -----------------------
# Run all exercises
# -----------------------
if __name__ == "__main__":
    exercise1_pytorch_basic()
    exercise2_pytorch_augmentation()
    exercise3_pytorch_pretrained()
    exercise4_pytorch_large()
