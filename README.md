# Handwriting-Predictive-A.I

A FastAPI-based RESTful API that identifies a person's handwriting using a trained PyTorch ResNet18 model. Upload a handwriting image and get the predicted writer with confidence scores. This project is aimed at combating exam malpractice by verifying handwriting.

## 🚀 Tech Stack

- **Backend:** FastAPI
- **Machine Learning:** PyTorch, Torchvision
- **Image Processing:** Pillow
- **Server:** Uvicorn

## 📂 Project Structure

```
.
├── app/
│   ├── main.py         # FastAPI application, endpoints
│   ├── model.py        # Model loading and prediction logic
│   └── utils.py        # Image preprocessing utilities
├── data/
│   └── handwriting_dataset_training/ # Training images
├── models/
│   └── handwriting_model.pt # Trained PyTorch model
├── .gitignore
├── README.md
├── requirements.txt    # Project dependencies
└── train.py            # Script to train the model
```

## ⚙️ Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Handwriting-Predictive-A.I.git
    cd Handwriting-Predictive-A.I
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## USAGE

1.  **Run the FastAPI server:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`.

2.  **Use the prediction endpoint:**
    You can use tools like `curl` or the interactive API documentation at `http://127.0.0.1:8000/docs` to send a `POST` request to the `/predict` endpoint with an image file.

    **Example using `curl`:**
    ```bash
    curl -X POST -F "file=@/path/to/your/image.jpg" http://127.0.0.1:8000/predict
    ```

    **Sample Prediction Response:**
    ```json
    {
      "label": "Dubem_test",
      "confidence": 0.9873,
      "message": "This handwriting definitely belongs to Dubem_test."
    }
    ```

## 🧠 Training the Model

To train the model on your own dataset:

1.  **Prepare your dataset:**
    Organize your training images in the `data/handwriting_dataset_training` directory, with each subdirectory named after a class (e.g., `data/handwriting_dataset_training/new_person`).

2.  **Run the training script:**
    ```bash
    python train.py
    ```
    The script will train the model and save the updated `handwriting_model.pt` file in the `models/` directory. You can configure the training parameters (epochs, learning rate, etc.) at the top of the `train.py` file.
