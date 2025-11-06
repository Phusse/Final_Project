import os
import random
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# === Config ===
DATA_DIR = 'data/handwriting_dataset_training'
MODEL_PATH = 'models/handwriting_model.pt'
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# === Preprocessing: Grayscale, resize, normalize ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),       # 👈 Convert to 1 channel
    transforms.Resize((224, 224)),                     # 👈 Match model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])        # 👈 Match inference
])

# === Dataset and Dataloaders ===
dataset = ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

# Create a map from class index to a list of sample indices
indices_by_class = {idx: [] for idx in range(len(dataset.classes))}
for i, (_, class_idx) in enumerate(dataset.samples):
    indices_by_class[class_idx].append(i)

# Split the classes (people)
class_indices = list(range(len(dataset.classes)))
random.shuffle(class_indices)
split_point = int(0.8 * len(class_indices))
train_class_indices = class_indices[:split_point]
val_class_indices = class_indices[split_point:]

# Collect the sample indices for each set
train_indices = []
for idx in train_class_indices:
    train_indices.extend(indices_by_class[idx])

val_indices = []
for idx in val_class_indices:
    val_indices.extend(indices_by_class[idx])

# Create Subset datasets
from torch.utils.data import Subset
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Model Setup ===
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 👈 Grayscale
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Loss and Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
print("🚀 Starting training...\n")
for epoch in range(NUM_EPOCHS):
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

    avg_loss = running_loss / len(train_loader)
    
    # --- Validation Loop ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
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
    accuracy = 100 * correct / total
    print(f"📘 Epoch {epoch + 1}/{NUM_EPOCHS} | Training Loss: {avg_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")

# === Save Model ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")

# === Testing Loop ===
print("\n🧪 Starting testing...")
TEST_DATA_DIR = 'data/handwriting_dataset_testing'

test_dataset = ImageFolder(TEST_DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n🎉 Final Test Accuracy: {accuracy:.2f}%")
