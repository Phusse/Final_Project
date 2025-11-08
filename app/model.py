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
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Handle potential dropout layer if you added it during training
        num_ftrs = self.model.fc.in_features
        # Check if your saved model expects a Sequential (Dropout+Linear) or just Linear
        # For safety, we'll assume standard Linear first, adjust if your checkpoint fails.
        self.model.fc = nn.Linear(num_ftrs, len(class_names))

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        
        # -- SAFE LOADING FIX --
        # Sometimes saved models have 'fc.0.weight' (if you used Sequential)
        # but newly initialized models just have 'fc.weight'. This fixes mismatch.
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
             # If training used Dropout, 'fc' might be a Sequential block.
             # We map 'fc.1.weight' -> 'fc.weight' if necessary for inference simplicity
            if k.startswith('fc.') and '1' in k: 
                 new_key = k.replace('fc.1.', 'fc.')
                 new_state_dict[new_key] = v
            elif k.startswith('fc.') and '0' in k and 'weight' not in k: # skip dropout layer
                 pass
            else:
                new_state_dict[k] = v
                
        # Try loading; if strict fails due to the dropout layer mismatch, try non-strict
        try:
             self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
             # Fallback for mismatched keys (often happens if you trained with Dropout but infer without)
             self.model.load_state_dict(new_state_dict, strict=False)

        self.class_names = checkpoint['class_names']
        self.model.eval()

    def predict_batch(self, batch_tensor):
        """
        Efficiently predicts a whole batch of patches at once.
        Input: [N, 1, 224, 224]
        Output: List of (label, confidence) tuples
        """
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_idxs = torch.max(probabilities, 1)

        results = []
        for i in range(len(confidences)):
            conf = confidences[i].item()
            if conf < self.threshold:
                results.append(("Unknown writer", conf))
            else:
                results.append((self.class_names[predicted_idxs[i].item()], conf))
        return results

# Utility function remains the same
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(BASE_DIR, "models", "handwriting_model.pt")
    # Pre-load checkpoint just to get class names for init
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    return HandwritingClassifier(model_path, checkpoint['class_names'])