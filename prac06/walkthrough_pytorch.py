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

def exercise1_pytorch_basic():
    """
    Description: Trains and evaluates a simple CNN on the CIFAR-10 dataset using
                 both RMSprop and Adam optimizers, and plots their confusion matrices.
    """
    print("--- Running Exercise 1: Basic CNN with multiple optimizers ---")

    # 1. Data Loading and Preprocessing
    transform = transforms.ToTensor()
    train_dataset_full = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_indices, val_indices = train_test_split(
        range(len(train_dataset_full)), test_size=0.2, stratify=train_dataset_full.targets, random_state=42)
    train_subset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset_full, val_indices)

    batch_size = 32
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 2. Model Definition
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 4 * 4, 10) # Adjusted size after 3 pools from 32x32

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x) # 16x16
            x = F.relu(self.conv2(x))
            x = self.pool(x) # 8x8
            x = F.relu(self.conv3(x))
            x = self.pool(x) # 4x4
            x = x.view(-1, 128 * 4 * 4)
            x = self.fc1(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Training and Evaluation Function (nested inside exercise1)
    def train_and_evaluate_optimizers(optimizer_name, learning_rate=0.001, epochs=5):
        print(f"\n--- Training with {optimizer_name} ---")
        model = CNNModel().to(device)

        if optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer not supported. Use 'RMSprop' or 'Adam'.")

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Epoch {epoch + 1}, "
                  f"Loss: {running_loss / len(train_loader):.4f}, "
                  f"Validation Accuracy: {100 * correct / total:.2f}%")

        print(f"Evaluating with {optimizer_name} on test data...")
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Test Accuracy: {100 * test_accuracy:.2f}%")
        return cm

    # 4. Run for both optimizers
    cm_rms = train_and_evaluate_optimizers("RMSprop")
    cm_adam = train_and_evaluate_optimizers("Adam")

    # 5. Plotting Confusion Matrices
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_rms, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.title("Confusion Matrix (RMSprop)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_adam, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.title("Confusion Matrix (Adam)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plt.tight_layout()
    # plt.savefig('ex1_confusion_matrix_pytorch.png') # Uncomment to save
    plt.show()


def exercise2_pytorch_augmentation():
    """
    Description: Trains a simple CNN with data augmentation on CIFAR-10
                 using the Adam optimizer.
    """
    print("\n--- Running Exercise 2: CNN with Data Augmentation ---")
    
    # 1. Data Loading and Preprocessing with Augmentation
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    test_val_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset_full = CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    val_dataset_for_split = CIFAR10(root='./data', train=True, download=True, transform=test_val_transforms)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_val_transforms)

    train_indices, val_indices = train_test_split(
        range(len(train_dataset_full)), test_size=0.2, stratify=train_dataset_full.targets, random_state=42)
    
    train_subset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset_for_split, val_indices)

    batch_size = 32
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 2. Model Definition
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 4 * 4, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(-1, 128 * 4 * 4)
            x = self.fc1(x)
            return x

    # 3. Training and Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 10

    for epoch in range(num_epochs):
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
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch + 1}, "
              f"Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Accuracy: {100 * correct / total:.2f}%")

    # 4. Evaluation on Test Data
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest accuracy: {test_accuracy:.3f}")
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.title("Confusion Matrix of Test Samples (Augmented)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    # plt.savefig('ex2_confusion_matrix_pytorch_augmented.png')
    plt.show()


def exercise3_pytorch_pretrained():
    """
    Description: Fine-tuning a pre-trained VGG16 model in PyTorch.
    """
    print("\n--- Running Exercise 3: Fine-tuning a pre-trained VGG16 ---")

    # 1. Data Loading and Preprocessing for VGG16
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset_full = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_indices, val_indices = train_test_split(
        range(len(train_dataset_full)), test_size=0.2, stratify=train_dataset_full.targets, random_state=42)
    
    train_subset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset_full, val_indices)

    batch_size = 32
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 2. Model Loading, Freezing, and Modification
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in vgg16.features.parameters():
        param.requires_grad = False

    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg16.to(device)

    # 4. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg16.parameters(), lr=1e-4)
    num_epochs = 3
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(vgg16.state_dict())

    print("Training the classifier head...")
    for epoch in range(num_epochs):
        vgg16.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})

        vgg16.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = vgg16(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(vgg16.state_dict())
            torch.save(best_model_wts, 'fine_tuning.pth')
            print(f"Model saved! New best validation loss: {best_val_loss:.4f}")

    # 5. Evaluate the best model
    vgg16.load_state_dict(best_model_wts)
    vgg16.eval()
    test_corrects, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_corrects += (predicted == labels).sum().item()

    test_acc = test_corrects / test_total
    print(f"\nFinal Test accuracy: {test_acc:.3f}")


def exercise4_pytorch_large():
    """
    Description: A larger PyTorch network trained from scratch on CIFAR-10.
    """
    print("\n--- Running Exercise 4: Training a larger network from scratch ---")

    # 1. Data Loading and Preprocessing
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset_full = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_indices, val_indices = train_test_split(
        range(len(train_dataset_full)), test_size=0.2, stratify=train_dataset_full.targets, random_state=42)
    
    train_subset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset_full, val_indices)

    batch_size = 64
    epochs = 20 # Can be higher, will use early stopping
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # wandb.init( ... ) # Wandb initialization can be placed here

    # 2. Model Definition
    class LargeCNN(nn.Module):
        def __init__(self):
            super(LargeCNN, self).__init__()
            # Block 1
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.dropout1 = nn.Dropout(0.25)
            # Block 2
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.dropout2 = nn.Dropout(0.25)
            # Block 3
            self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn5 = nn.BatchNorm2d(128)
            self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
            self.bn6 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.dropout3 = nn.Dropout(0.25)
            # Final dense layers
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.dropout4 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool1(x)
            x = self.dropout1(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool2(x)
            x = self.dropout2(x)
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x = self.pool3(x)
            x = self.dropout3(x)
            x = x.view(x.size(0), -1) # Flatten
            x = F.relu(self.fc1(x))
            x = self.dropout4(x)
            x = self.fc2(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LargeCNN().to(device)
    
    # 3. Training and Evaluation
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        print(f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {100 * val_accuracy:.2f}%")
        
        # wandb.log({ ... }) # wandb logging can be placed here

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print("Early stopping triggered.")
            break

    # 4. Evaluate the best model
    model.load_state_dict(best_model_wts)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nFinal Test accuracy: {100 * test_accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix for Large CNN')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    # plt.savefig('ex4_confusion_matrix_large.png')
    plt.show()

def main():
    exercise1_pytorch_basic()
    exercise2_pytorch_augmentation()
    exercise3_pytorch_pretrained()
    exercise4_pytorch_large()

if __name__ == "__main__":
    main()