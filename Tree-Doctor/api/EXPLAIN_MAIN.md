# Understanding `main.py` with Code Snippets

This file is the **engine** of the application. It creates the web server and handles the AI predictions.

---

### 1. Setting the Stage (Imports & Mock Mode)
We import `FastAPI` to build the web server and `tensorflow` to run the AI model. We use a `try-except` block to ensure the app doesn't crash if TensorFlow is missing (Mock Mode).

```python
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed. Running in Mock Mode.")
    tf = None

app = FastAPI()
```

---

### 2. Connecting to the Website (CORS)
We need to allow the frontend (running on `localhost:3000`) to talk to this backend.

```python
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### 3. Loading the AI Model
We look for the trained model file. If found, we load it into memory.

```python
MODEL_PATH = os.path.join(BASE_DIR, "models", "potato_model.keras")

if tf and os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
else:
    model = None
    print("Model not found. Predictions will be mocked.")
```

---

### 4. The "Prediction" Endpoint
When you upload an image, it comes here. We convert the image file into numbers (numpy array) that the model can understand.

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Convert uploaded file to image
    image = read_file_as_image(await file.read())
```

**If the Model works:**
We resize the image to 256x256 (what the model expects) and ask for a prediction.

```python
    if tf:
        image = tf.image.resize(image, [256, 256])
        img_batch = np.expand_dims(image, 0)
        predictions = model.predict(img_batch)
        
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
```

**If we are in Mock Mode:**
We pretend to predict based on the filename (just for testing).

```python
    if not model:
        if "early" in file.filename.lower():
             predicted_class = "Potato___Early_blight"
        # ... (logic to set confidence)
```

---

### 5. Returning the Result
Finally, we attach some medical advice and send it back to the user.

```python
    # Logic to select recommendations based on predicted_class
    if predicted_class == "Potato___Early_blight":
        recommendations = {
             "medicines": ["Mancozeb", "Chlorothalonil"],
             "dosage": "2–2.5 g per liter",
             # ...
        }

    return {
        "class": predicted_class,
        "confidence": float(confidence),
        "recommendations": recommendations
    }
```
