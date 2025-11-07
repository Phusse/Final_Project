import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# === CONFIG ===
DATA_DIR = 'data/handwriting_dataset_training'
TEST_DATA_DIR = 'data/handwriting_dataset_testing'
MODEL_PATH = 'models/handwriting_model.pt'

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2  # 80% train / 20% validation

# === PREPROCESSING ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # Convert to grayscale
    transforms.Resize((224, 224)),                 # Resize for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize to match model expectations
])

# === DATASET SETUP ===
dataset = ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

# Split into train and validation (sample-based split)
train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📂 Loaded dataset with {num_classes} writers")
print(f"📘 Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

# === MODEL SETUP ===
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Grayscale support
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === LOSS + OPTIMIZER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === TRAINING LOOP ===
print("\n🚀 Starting training...\n")

for epoch in range(NUM_EPOCHS):
    # ---- TRAIN ----
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

    avg_train_loss = running_loss / len(train_loader)

    # ---- VALIDATION ----
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

    print(f"📘 Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

# === SAVE MODEL ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")

# === TESTING LOOP ===
if os.path.exists(TEST_DATA_DIR):
    print("\n🧪 Starting testing...")

    test_dataset = ImageFolder(TEST_DATA_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    test_accuracy = 100 * correct / total
    print(f"\n🎉 Final Test Accuracy: {test_accuracy:.2f}%")
else:
    print("\n⚠️ Skipping test phase — no testing folder found.")
