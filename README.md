# Hate Speech Detection

A Deep Learning system built with **TensorFlow / Keras** and **FastAPI** to classify social media text into **Hate Speech**, **Offensive Language**, and **Neutral**.

---

## 🚀 Features

- **Bidirectional LSTM Model**: Trained on the Davidson et al. Twitter dataset with word embeddings and sequence padding.
- **CLI Predictor**: Instant terminal prediction script with full 3-class probability breakdowns.
- **REST API & Interactive Web Interface**: Built with **FastAPI** & Tailwind CSS (`main.py`) with live visual progress bars and Swagger docs (`/docs`).
- **Pretrained Weights & Tokenizer**: Includes `hatespeechdetector.keras` and reconstructed `hatespeechdetector_tokenizer.pkl`.

---

## 💻 Running the Project

### Option A: Using the Convenient Batch Launchers (Easiest)

From the project folder in PowerShell or Command Prompt:

```cmd
# Run CLI prediction on any text
.\predict.bat "Have a wonderful and blessed day everyone!"

# Launch the Web App and REST API
.\run_web.bat
```
Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

### Option B: Using the Virtual Environment (`.venv`)

Activate the pre-configured virtual environment:

**In PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
python predict.py "Have a wonderful and blessed day everyone!"
python main.py
```

*Or invoke the virtual environment's Python directly without activating:*
```powershell
.\.venv\Scripts\python predict.py "Have a wonderful and blessed day everyone!"
.\.venv\Scripts\python main.py
```

---

### Option C: Using Anaconda's Python

```powershell
C:\Users\Shreyash\anaconda3\python.exe predict.py "Have a wonderful and blessed day everyone!"
C:\Users\Shreyash\anaconda3\python.exe main.py
```

---

## 📊 Model Architecture

- **Embedding Layer**: 5,000 words vocabulary, 32-dim embeddings
- **Bidirectional LSTM**: 16 units with forward and backward directions
- **Dense Classifier**: 512 units (ReLU, L1 regularization) + BatchNormalization + Dropout (0.3)
- **Output Layer**: 3 classes with Softmax activation
  - `0`: Hate Speech
  - `1`: Offensive Language
  - `2`: Neutral
