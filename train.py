import os
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
NUM_EPOCHS = 10
BATCH_SIZE = 16
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

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
    print(f"📘 Epoch {epoch + 1}/{NUM_EPOCHS} | Training Loss: {avg_loss:.4f}")

# === Save Model ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), 'models/handwriting_model.pt')
print(f"\n✅ Model saved to {MODEL_PATH}")
