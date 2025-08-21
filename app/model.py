import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class HandwritingClassifier:
    def __init__(self, model_path, class_names, threshold=0.85):
        self.class_names = class_names
        self.threshold = threshold

        # ✅ Load a ResNet18 model with pretrained weights
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # ✅ Modify the first conv layer to accept 1-channel grayscale input
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # ✅ Modify the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, len(class_names))

        # ✅ Load model weights
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def predict(self, input_tensor):
        """Predict the class of a handwriting image tensor"""
        with torch.no_grad():
            outputs = self.model(input_tensor)

            # ✅ Apply softmax to get class probabilities
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            confidence = confidence.item()

            if confidence < self.threshold:
                return "Unknown writer", confidence

            label = self.class_names[predicted_idx.item()]
            return label, confidence


# ✅ Utility function to load the model with correct class labels
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(BASE_DIR, "models", "handwriting_model.pt")

    class_names = ["Agino", "Christabel", "Dubem", "Goodness", "David_Lee", "eshiet", "John_Martin", "Thompson"]
    return HandwritingClassifier(model_path, class_names)
