from PIL import Image
import pytesseract
import os
import random

INPUT_IMAGE = "IMG-20250804-WA0100.jpg"
OUTPUT_DIR = r"C:\Users\TopBoy\Desktop\Final_Project\data\handwriting_dataset_testing\juliet\jay3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Add this line
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open(INPUT_IMAGE)
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

words = []
for i, text in enumerate(data["text"]):
    if text.strip():
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append((x, y, w, h))

i = 0
group_index = 1
while i < len(words):
    group_size = random.randint(1, 4)
    group = words[i:i+group_size]

    x0 = min([w[0] for w in group])
    y0 = min([w[1] for w in group])
    x1 = max([w[0] + w[2] for w in group])
    y1 = max([w[1] + w[3] for w in group])

    cropped = img.crop((x0, y0, x1, y1))
    cropped.save(os.path.join(OUTPUT_DIR, f"word_{group_index}.jpg"))
    group_index += 1
    i += group_size

print(f"✅ Done! Cropped {group_index-1} word-groups saved to {OUTPUT_DIR}")
