import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import models

# === CONFIG ===
DATA_DIR = 'data/handwriting_dataset_training'
TEST_DATA_DIR = 'data/handwriting_dataset_testing'
MODEL_PATH = 'models/handwriting_model.pt'

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2

# === PREPROCESSING & AUGMENTATION ===
transform_train = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2), translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_eval = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === DATASET SPLIT ===
dataset = ImageFolder(DATA_DIR, transform=transform_train)
class_names = dataset.classes
num_classes = len(class_names)

train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Validation set uses evaluation transforms
val_dataset.dataset.transform = transform_eval

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📂 Loaded dataset with {num_classes} writers")
print(f"📘 Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

# === MODEL SETUP ===
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# This adds a 50% chance for neurons to be turned off during training
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === TRAIN IN TWO PHASES: FREEZE THEN UNFREEZE ===
# Phase 1: train classifier head only
for name, param in model.named_parameters():
    if not name.startswith('fc'):
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

best_val_acc = 0.0
print("\n🚀 Starting Phase 1: training classification head...\n")

for epoch in range(NUM_EPOCHS // 2):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    scheduler.step(acc)
    print(f"📘 Epoch {epoch+1:02}/{NUM_EPOCHS//2} | Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Accuracy: {acc:.2f}%")

    if acc > best_val_acc:
        best_val_acc = acc
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, MODEL_PATH)

# === Phase 2: Unfreeze backbone for fine-tuning ===
print("\n🔓 Unfreezing backbone for fine-tuning...\n")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

for epoch in range(NUM_EPOCHS // 2, NUM_EPOCHS):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    scheduler.step(acc)
    print(f"📘 Epoch {epoch+1:02}/{NUM_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Accuracy: {acc:.2f}% | Best: {best_val_acc:.2f}%")

    if acc > best_val_acc:
        best_val_acc = acc
        torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, MODEL_PATH)

print(f"\n✅ Best model saved to {MODEL_PATH} | Final Val Accuracy: {best_val_acc:.2f}%")

# === TESTING ===
if os.path.exists(TEST_DATA_DIR):
    print("\n🧪 Starting testing...")
    test_dataset = ImageFolder(TEST_DATA_DIR, transform=transform_eval)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"\n🎉 Final Test Accuracy: {test_acc:.2f}%")
else:
    print("\n⚠️ No testing folder found, skipping test phase.")
