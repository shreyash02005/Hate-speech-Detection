import os
import sys

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import keras
import numpy as np
import pickle
from pathlib import Path

# Paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hatespeechdetector.keras"
TOKENIZER_PATH = BASE_DIR / "hatespeechdetector_tokenizer.pkl"

CLASS_NAMES = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral"
}

# Load model and tokenizer
print("Loading model and tokenizer...")
model = keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("Ready!")

def predict(text: str):
    sequence = tokenizer.texts_to_sequences([text])
    padded = keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=100, padding="post", truncating="post"
    )
    prediction = model.predict(padded, verbose=0)[0]
    
    top_class_id = int(np.argmax(prediction))
    top_label = CLASS_NAMES[top_class_id]
    
    scores = {
        CLASS_NAMES[i]: float(prediction[i]) for i in range(len(prediction))
    }
    
    # Binary classification logic: Is it hate speech?
    is_hate_speech = bool(prediction[0] > 0.5)
    
    return {
        "text": text,
        "label": top_label,
        "is_hate_speech": is_hate_speech,
        "confidence": float(prediction[top_class_id]),
        "class_probabilities": scores
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "Have a wonderful and blessed day everyone!"
    
    result = predict(input_text)
    print("\n--- Prediction Result ---")
    print(f"Text:           {result['text']}")
    print(f"Top Label:      {result['label']}")
    print(f"Confidence:     {result['confidence']:.2%}")
    print(f"Hate Speech:    {'Yes' if result['is_hate_speech'] else 'No'}")
    print("Class Probabilities:")
    for cls, prob in result["class_probabilities"].items():
        print(f"  - {cls:<20}: {prob:.2%}")
