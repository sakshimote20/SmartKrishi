# ml_models/predictor.py

import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer

# --- Dummy Layer Class (used in your model) ---
class DummyCast(Layer):
    def __init__(self, dtype=None, **kwargs):
        super(DummyCast, self).__init__(**kwargs)
        self.dtype_ = dtype
    def call(self, inputs):
        return inputs

# --- Load class indices ---
with open("ml_models/class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}
class_names = [index_to_class[i] for i in range(len(index_to_class))]

# --- Load treatments ---
with open("ml_models/treatments_full_76.json", "r") as f:
    treatment_data = json.load(f)

# --- Load Model ---
custom_objects = {'Cast': DummyCast}
with custom_object_scope(custom_objects):
    model = tf.keras.models.load_model("ml_models/cropshield_ai.h5", compile=False)

# --- Predict function ---
def predict_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    result = class_names[predicted_index]

    # Get treatment
    treatment = treatment_data.get(result, {
        "chemical": "No recommendation available.",
        "biological": "No recommendation available."
    })

    return {
        "disease": result,
        "confidence": f"{confidence:.2f}%",
        "chemical_treatment": treatment["chemical"],
        "biological_treatment": treatment["biological"]
    }
