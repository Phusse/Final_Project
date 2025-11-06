import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_names']

def predict_handwriting(image_path, model, class_names, threshold=0.7):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    image = Image.open(image_path).convert('L')
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        confidence = confidence.item()
        writer_name = class_names[predicted.item()]

    if confidence < threshold:
        return {"match": False, "confidence": confidence}
    else:
        return {"match": True, "writer": writer_name, "confidence": confidence}
