from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import io
import torch
from collections import Counter
from app.model import load_model
from app.utils import smart_shred, preprocess_batch

app = FastAPI()
model = load_model()

@app.post("/predict")
async def predict_handwriting(file: UploadFile = File(...)):
    # Step 1: Basic Validation
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, detail="Invalid file type. Only .jpg and .png allowed.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image)
    except:
        raise HTTPException(400, detail="Invalid image file.")

    # Step 2: THE SMART UPGRADE - Shred the image
    # This will return a list of small, clear images of words/lines
    patches = smart_shred(image, max_patches=15)
    
    if not patches:
         raise HTTPException(400, detail="Could not find any clear handwriting on this page.")

    # Step 3: Batch process all patches at once
    batch_tensor = preprocess_batch(patches)
    predictions = model.predict_batch(batch_tensor)
    
    # Step 4: Hold the Vote
    # predictions is a list like: [('Mike', 0.9), ('Sarah', 0.88), ('Unknown', 0.6)]
    votes = [pred[0] for pred in predictions]
    vote_counts = Counter(votes)
    
    # Find the winner
    winner, count = vote_counts.most_common(1)[0]
    
    # Calculate average confidence only for the winner's patches
    winner_confidences = [p[1] for p in predictions if p[0] == winner]
    avg_confidence = sum(winner_confidences) / len(winner_confidences) if winner_confidences else 0.0

    # Step 5: Final Verdict Logic
    # If 'Unknown' won the vote, return Unknown.
    # If the winner didn't get at least 50% of the total patches, might be too unsure.
    total_patches = len(patches)
    if winner == "Unknown writer" or (count / total_patches < 0.4): # mild threshold for majority
         final_label = "Unknown writer"
         final_message = f"Ambiguous document. Best guess was {winner}, but only found in {count}/{total_patches} regions."
    else:
         final_label = winner
         final_message = f"Document recognized as {winner} (matched {count}/{total_patches} regions analyzed)."

    return JSONResponse(status_code=200, content={
        "label": final_label,
        "confidence": round(avg_confidence, 4),
        "message": final_message,
        "debug_votes": vote_counts # Helpful to see seeing split decisions
    })