import os
import sys
from pathlib import Path
import warnings

# Suppress noisy TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import keras
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hatespeechdetector.keras"
TOKENIZER_PATH = BASE_DIR / "hatespeechdetector_tokenizer.pkl"

CLASS_NAMES = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral"
}

# Initialize FastAPI App
app = FastAPI(
    title="Hate Speech Detection API",
    description="Deep Learning API for classifying text into Hate Speech, Offensive Language, and Neutral.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer lazily / at startup
model = None
tokenizer = None

def get_model_and_tokenizer():
    global model, tokenizer
    if model is None or tokenizer is None:
        if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
            raise RuntimeError("Model or Tokenizer file not found.")
        model = keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    return model, tokenizer

@app.on_event("startup")
def startup_event():
    get_model_and_tokenizer()

class PredictRequest(BaseModel):
    text: str = Field(..., description="Text string to evaluate", min_length=1)

class PredictResponse(BaseModel):
    text: str
    label: str
    is_hate_speech: bool
    confidence: float
    probabilities: dict[str, float]

def classify_text(text: str) -> dict:
    mdl, tok = get_model_and_tokenizer()
    sequence = tok.texts_to_sequences([text])
    padded = keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=100, padding="post", truncating="post"
    )
    prediction = mdl.predict(padded, verbose=0)[0]
    top_class_id = int(np.argmax(prediction))
    top_label = CLASS_NAMES[top_class_id]
    
    probabilities = {
        CLASS_NAMES[i]: round(float(prediction[i]), 4) for i in range(len(prediction))
    }
    
    return {
        "text": text,
        "label": top_label,
        "is_hate_speech": bool(prediction[0] > 0.5),
        "confidence": round(float(prediction[top_class_id]), 4),
        "probabilities": probabilities
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return classify_text(payload.text)

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hate Speech Detector</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col justify-between">
  <div class="max-w-3xl mx-auto px-4 py-12 w-full">
    <!-- Header -->
    <div class="text-center mb-8">
      <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-400 text-sm font-semibold mb-3 border border-indigo-500/20">
        <span class="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"></span>
        Bidirectional LSTM Classifier
      </div>
      <h1 class="text-4xl font-extrabold tracking-tight text-white mb-2">Hate Speech Detection</h1>
      <p class="text-slate-400">Classify text into <span class="text-rose-400 font-medium">Hate Speech</span>, <span class="text-amber-400 font-medium">Offensive Language</span>, or <span class="text-emerald-400 font-medium">Neutral</span>.</p>
    </div>

    <!-- Main Card -->
    <div class="bg-slate-800/80 backdrop-blur border border-slate-700/60 rounded-2xl shadow-xl p-6 sm:p-8">
      <label for="inputText" class="block text-sm font-medium text-slate-300 mb-2">Enter text to evaluate</label>
      <textarea id="inputText" rows="4" class="w-full bg-slate-900/90 border border-slate-700 rounded-xl p-4 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition resize-none" placeholder="Type or paste a message/tweet here..."></textarea>
      
      <!-- Quick samples -->
      <div class="mt-3 flex flex-wrap items-center gap-2 text-xs">
        <span class="text-slate-400">Try sample:</span>
        <button onclick="setSample('neutral')" class="px-2.5 py-1 rounded-lg bg-slate-700/60 hover:bg-slate-700 text-slate-300 transition">Clean tweet</button>
        <button onclick="setSample('offensive')" class="px-2.5 py-1 rounded-lg bg-slate-700/60 hover:bg-slate-700 text-slate-300 transition">Offensive</button>
        <button onclick="setSample('hate')" class="px-2.5 py-1 rounded-lg bg-slate-700/60 hover:bg-slate-700 text-slate-300 transition">Hate speech</button>
      </div>

      <!-- Action Button -->
      <div class="mt-6 flex justify-end">
        <button id="analyzeBtn" onclick="runAnalysis()" class="w-full sm:w-auto px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-semibold text-white shadow-lg shadow-indigo-600/30 transition flex items-center justify-center gap-2">
          <svg id="spinner" class="animate-spin -ml-1 mr-2 h-4 w-4 text-white hidden" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
          </svg>
          <span id="btnText">Analyze Text</span>
        </button>
      </div>

      <!-- Results Area -->
      <div id="resultContainer" class="hidden mt-8 pt-6 border-t border-slate-700/80">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Classification Result</h2>
          <span id="labelBadge" class="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide"></span>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
          <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-700/40">
            <div class="text-xs text-rose-400 font-medium mb-1">Hate Speech</div>
            <div id="probHate" class="text-2xl font-bold text-slate-100">0%</div>
            <div class="w-full bg-slate-800 rounded-full h-1.5 mt-2 overflow-hidden">
              <div id="barHate" class="bg-rose-500 h-1.5 transition-all duration-500" style="width: 0%"></div>
            </div>
          </div>

          <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-700/40">
            <div class="text-xs text-amber-400 font-medium mb-1">Offensive Language</div>
            <div id="probOffensive" class="text-2xl font-bold text-slate-100">0%</div>
            <div class="w-full bg-slate-800 rounded-full h-1.5 mt-2 overflow-hidden">
              <div id="barOffensive" class="bg-amber-500 h-1.5 transition-all duration-500" style="width: 0%"></div>
            </div>
          </div>

          <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-700/40">
            <div class="text-xs text-emerald-400 font-medium mb-1">Neutral</div>
            <div id="probNeutral" class="text-2xl font-bold text-slate-100">0%</div>
            <div class="w-full bg-slate-800 rounded-full h-1.5 mt-2 overflow-hidden">
              <div id="barNeutral" class="bg-emerald-500 h-1.5 transition-all duration-500" style="width: 0%"></div>
            </div>
          </div>
        </div>

        <div class="bg-slate-900/40 rounded-xl p-4 border border-slate-700/40 text-xs text-slate-400 flex flex-col gap-1">
          <div><strong class="text-slate-300">Model:</strong> Bidirectional LSTM with 100-dim sequence padding</div>
          <div><strong class="text-slate-300">API Documentation:</strong> <a href="/docs" target="_blank" class="text-indigo-400 hover:underline">Interactive Swagger UI (/docs)</a></div>
        </div>
      </div>
    </div>
  </div>

  <footer class="py-6 text-center text-xs text-slate-500">
    Hate Speech Detection &bull; Powered by TensorFlow, Keras & FastAPI
  </footer>

  <script>
    const samples = {
      neutral: "I really enjoyed reading this book, highly recommended to everyone!",
      offensive: "Shut up you stupid bitch, stop acting like an idiot.",
      hate: "I hate these dirty beaners and queers"
    };

    function setSample(type) {
      document.getElementById('inputText').value = samples[type] || '';
    }

    async function runAnalysis() {
      const text = document.getElementById('inputText').value.trim();
      if (!text) return;

      const btn = document.getElementById('analyzeBtn');
      const btnText = document.getElementById('btnText');
      const spinner = document.getElementById('spinner');
      const resContainer = document.getElementById('resultContainer');

      btn.disabled = true;
      spinner.classList.remove('hidden');
      btnText.innerText = 'Analyzing...';

      try {
        const resp = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        const data = await resp.json();

        resContainer.classList.remove('hidden');

        const badge = document.getElementById('labelBadge');
        badge.innerText = data.label;
        badge.className = 'px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide ';
        if (data.label === 'Hate Speech') {
          badge.className += 'bg-rose-500/20 text-rose-300 border border-rose-500/30';
        } else if (data.label === 'Offensive Language') {
          badge.className += 'bg-amber-500/20 text-amber-300 border border-amber-500/30';
        } else {
          badge.className += 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
        }

        const hPct = (data.probabilities['Hate Speech'] * 100).toFixed(1) + '%';
        const oPct = (data.probabilities['Offensive Language'] * 100).toFixed(1) + '%';
        const nPct = (data.probabilities['Neutral'] * 100).toFixed(1) + '%';

        document.getElementById('probHate').innerText = hPct;
        document.getElementById('probOffensive').innerText = oPct;
        document.getElementById('probNeutral').innerText = nPct;

        document.getElementById('barHate').style.width = hPct;
        document.getElementById('barOffensive').style.width = oPct;
        document.getElementById('barNeutral').style.width = nPct;

      } catch (err) {
        alert('Analysis error: ' + err.message);
      } finally {
        btn.disabled = false;
        spinner.classList.add('hidden');
        btnText.innerText = 'Analyze Text';
      }
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
