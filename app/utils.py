from PIL import Image
import torchvision.transforms as transforms

# Transformation pipeline for grayscale ResNet model
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure 1-channel input
    transforms.Resize((224, 224)),                # Match ResNet input size
    transforms.ToTensor(),                        # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize as trained
])

def preprocess_image(image: Image.Image):
    """
    Preprocess a PIL image for grayscale ResNet input.
    Ensures image is grayscale, resized, normalized, and batched.
    """
    image = image.convert('L')  # Convert to grayscale ('L' mode)
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension (1, 1, 224, 224)
