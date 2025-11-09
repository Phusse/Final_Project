import os
import cv2
import math
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
INPUT_DIR = Path(r'C:\Users\HP\Documents\Final_Project\data\handwriting_dataset_testing') # Where your 80 images are now
OUTPUT_DIR = Path(r'C:\Users\HP\Documents\Final_Project\data\handwriting_dataset_testing\ready_for_training')          # New folder for fixed data

def make_square_patches(img_path, output_folder):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return

    h, w = img.shape
    
    # If it's already mostly square (aspect ratio between 0.5 and 2), just save it resized
    if 0.5 < w/h < 2.0:
        # Resize to exactly 224x224 to be safe
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_folder / img_path.name), resized)
        return

    # If it's a long strip, chop it into square-ish chunks
    # How many chunks? Width divided by Height, rounded up.
    num_patches = math.ceil(w / h)
    patch_width = math.ceil(w / num_patches)

    for i in range(num_patches):
        start_x = i * patch_width
        end_x = min(start_x + patch_width, w)
        
        # Extract patch
        patch = img[0:h, start_x:end_x]
        
        # If it's too skinny now, pad it with white to make it square
        ph, pw = patch.shape
        if pw < ph:
            pad_l = (ph - pw) // 2
            pad_r = ph - pw - pad_l
            patch = cv2.copyMakeBorder(patch, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=255)
            
        # Resize to standard 224x224
        final_patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Save with a new name
        new_name = f"{img_path.stem}_p{i}.jpg"
        cv2.imwrite(str(output_folder / new_name), final_patch)

# === RUN IT ===
for class_dir in INPUT_DIR.iterdir():
    if class_dir.is_dir():
        target_dir = OUTPUT_DIR / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"🔧 Fixing images for {class_dir.name}...")
        for img_file in class_dir.glob('*'):
             if img_file.suffix.lower() in ['.jpg', '.png']:
                 make_square_patches(img_file, target_dir)

print("\n✅ Done! Point your training script to 'data/ready_for_training' now.")