# Final_Project - A handwriting predictive A.I algorithm for combating exam malpractice.

A FastAPI-based RESTful API that identifies a person's handwriting using a trained PyTorch ResNet18 model. Upload a handwriting image and get the predicted writer with confidence scores.

---

## 🚀 Features

- ✅ Predicts handwriting from `.jpg` or `.png` images
- ✅ Uses a custom-trained ResNet18 deep learning model
- ✅ Preprocessing includes grayscale conversion, resizing, and normalization
- ✅ Returns prediction label, confidence score, and interpretation message
- ✅ Handles image rotation and validation automatically

---

## 🖼️ Sample Prediction Response

```json
{
  "label": "Dubem_test",
  "confidence": 0.9873,
  "message": "This handwriting definitely belongs to Dubem_test."
}
