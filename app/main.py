from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import imghdr
import io
import torch
from app.model import load_model
from app.utils import preprocess_image

app = FastAPI()

# Load trained handwriting classifier model
model = load_model()

# Confidence thresholds for human-friendly messaging
CONFIDENCE_THRESHOLDS = {
    "certain": 0.80,
    "likely": 0.70
}

@app.post("/predict")
async def predict_handwriting(file: UploadFile = File(...)):
    # Step 1: Validate file type
    contents = await file.read()
    image_type = imghdr.what(None, h=contents)
    if image_type not in ["jpeg", "png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only .jpg and .png allowed.")

    # Step 2: Open and correct the image
    try:
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image)  # Correct orientation from EXIF
        if image.width > image.height:
            image = image.rotate(-90, expand=True)  # Optional heuristic
        image = image.convert('L')  # Ensure grayscale
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image. Ensure it’s a valid image file.")

    # Step 3: Check resolution
    if image.width < 100 or image.height < 100:
        raise HTTPException(status_code=400, detail="Image is too small. Please upload a clearer image.")

    # Step 4: Preprocess image and predict
    input_tensor = preprocess_image(image)

    # Double-check shape
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dim

    label, confidence = model.predict(input_tensor)

    # Clamp confidence to [0, 1]
    confidence = float(torch.clamp(torch.tensor(confidence), 0.0, 1.0).item())

    # Step 5: Generate interpretation
    if confidence >= CONFIDENCE_THRESHOLDS["certain"]:
        message = f"This handwriting definitely belongs to {label}."
    elif confidence >= CONFIDENCE_THRESHOLDS["likely"]:
        message = f"This handwriting likely belongs to {label}."
    else:
        label = "Unknown writer"
        message = "This handwriting is not recognized. Likely an impersonator or someone not in the training set."

    # Step 6: Return result
    return JSONResponse(
        status_code=200,
        content={
            "label": label,
            "confidence": round(confidence, 4),
            "message": message
        }
    )
