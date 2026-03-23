import keras
import numpy as np
import pickle

# Load model
model = keras.models.load_model("hatespeechdetector.keras")

# Load tokenizer
with open("hatespeechdetector_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100) # use your training maxlen
    prediction = model.predict(padded)
    label = "Hate Speech" if prediction[0][0] > 0.5 else "Not Hate Speech"
    confidence = float(prediction[0][0])
    return {"label": label, "confidence": confidence}

# Testing
print(predict("your test sentence here"))