from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import pickle
import os
import contextlib

from utils.preprocessing import clean_text
from utils.explain import XAIExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Global variables for loaded artifacts
model = None
tokenizer = None
explainer = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'model', 'tokenizer.pkl')
MAX_LENGTH = 50

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, explainer
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        explainer = XAIExplainer(model, tokenizer, max_length=MAX_LENGTH)
        print("Model and Tokenizer loaded successfully.")
    else:
        print("Warning: Model or Tokenizer not found! Please run train.py.")
    yield
    # Cleanup code here if needed

app = FastAPI(title="Fake News Detection API with XAI", lifespan=lifespan)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_news(req: PredictionRequest):
    if not model or not tokenizer or not explainer:
        raise HTTPException(status_code=500, detail="Model is not loaded. Ensure train.py has been run.")

    original_text = req.text.strip()
    if not original_text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # 1. Preprocess the text to get sequence
    cleaned = clean_text(original_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

    # 2. Get Prediction
    # model pred is probability of REAL
    pred_prob = model.predict(padded, verbose=0)[0][0] 
    
    # Threshold at 0.5
    prediction_label = "REAL" if pred_prob >= 0.5 else "FAKE"
    
    # If it's REAL, confidence is pred_prob. If FAKE, confidence is 1 - pred_prob
    confidence = float(pred_prob) if prediction_label == "REAL" else float(1.0 - pred_prob)
    
    # 3. Get Explainability (LIME)
    # LIME expects raw text, the wrapper handles preprocessing inside predict_proba
    explanation = explainer.explain(original_text)

    # Format explanation for frontend ease
    # explanation looks like: [('word1', 0.12), ('word2', -0.05), ...]
    # For LIME within our wrapper:
    # class 1 = Real. So positive weights -> towards Real. Negative weights -> towards Fake.
    formatted_explanation = []
    for word, weight in explanation:
        indicator = "REAL" if weight > 0 else "FAKE"
        formatted_explanation.append({
            "word": word,
            "weight": float(weight),
            "indicator": indicator
        })

    return {
        "prediction": prediction_label,
        "confidence": round(confidence * 100, 2), # percentage
        "explanation": formatted_explanation
    }
