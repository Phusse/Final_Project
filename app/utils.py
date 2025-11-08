from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np

# Transformation pipeline for grayscale ResNet model
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(image: Image.Image):
    """Standard generic preprocessor for single images."""
    image = image.convert('L')
    return transform(image).unsqueeze(0)

def smart_shred(image: Image.Image, max_patches=20):
    """
    Smartly shreds a full page into smaller, square-ish patches containing handwriting.
    """
    # Convert PIL to OpenCV format
    cv_img = np.array(image.convert('L'))

    # 1. Threshold to find ink (invert so ink is white on black background)
    _, thresh = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Dilate (thicken) ink to merge nearby letters into words/lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)) # Wide kernel for horizontal text
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # 3. Find contours (blobs of text)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Filter and sort contours by area (largest first)
    min_area = 2000  # Ignore small noise dots
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:max_patches]

    patches = []
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        # Add a little padding around the cut
        pad = 10
        x, y = max(0, x-pad), max(0, y-pad)
        w, h = min(cv_img.shape[1]-x, w+2*pad), min(cv_img.shape[0]-y, h+2*pad)

        # Crop and convert back to PIL
        patch = Image.fromarray(cv_img[y:y+h, x:x+w])
        patches.append(patch)

    # Fallback: If no good contours found, just return original image as 1 patch
    if not patches:
        return [image.convert('L')]

    return patches

def preprocess_batch(patches: list):
    """
    Takes a list of PIL patches and stacks them into one big tensor.
    Returns tensor shape: [NUM_PATCHES, 1, 224, 224]
    """
    tensors = [transform(patch.convert('L')) for patch in patches]
    return torch.stack(tensors)