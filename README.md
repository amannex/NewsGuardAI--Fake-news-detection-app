# NewsGuard AI: Fake News Detection using Explainable AI (XAI)

NewsGuard AI is an end-to-end web platform and browser extension designed to detect fake news using Deep Learning. Going beyond a simple "Fake" or "Real" binary classification, this system utilizes **Explainable AI (LIME)** to visualize *why* the model made its decision, highlighting specific words that influenced the prediction.

## 🌟 Key Features
- **Deep Learning Core**: Built on a Bidirectional LSTM (BiLSTM) model trained on the ISOT Fake News Dataset (~45,000 articles).
- **Explainable AI (XAI)**: Uses LIME (Local Interpretable Model-Agnostic Explanations) to provide human-readable visualizations of the model's decision-making process.
- **Modern Web Interface**: A beautifully designed, responsive Light Theme frontend with real-time text analysis.
- **Browser Extension**: Includes a Chrome/Edge extension that allows users to seamlessly analyze text highlighted on any webpage.

## 🛠️ Tech Stack
- **Backend**: Python, FastAPI, TensorFlow/Keras, LIME, Pandas, NLTK
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (No complex build tools required)
- **Extension**: Chrome Manifest V3, HTML/JS/CSS
- **Model**: Bidirectional LSTM (BiLSTM)
- **Dataset**: ISOT Fake News Dataset

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. 

### 2. Backend Setup
1. Open your terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Model Training (Optional, if you need to retrain)
To train the model on the full real-world dataset:
1. Download the **ISOT Fake News Dataset** (often listed as "Fake and real news dataset" on Kaggle).
2. Place the extracted `Fake.csv` and `True.csv` inside a newly created `backend/data/` folder.
3. Run the training script:
   ```bash
   python model/train.py
   ```
   *(This will read all ~45,000 articles, train the BiLSTM, and save `model.h5` and `tokenizer.pkl` to the `backend/model` folder).*

### 4. Running the Application
1. **Start the FastAPI Backend**:
   Ensure your virtual environment is activated, then run:
   ```bash
   uvicorn app:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.

2. **Serve the Frontend**:
   Open a new terminal, navigate to the `frontend/` directory, and run a simple HTTP server:
   ```bash
   python -m http.server 8080
   ```
   Visit `http://localhost:8080` in your browser to view the app!

---

## 🧩 Installing the Browser Extension
1. Open Google Chrome or Microsoft Edge.
2. Navigate to `chrome://extensions/` (or `edge://extensions/`).
3. Enable **Developer mode** in the top right corner.
4. Click **Load unpacked** and select the `extension/` folder located in this repository.
5. The NewsGuard AI icon will appear in your browser. Click it to paste and analyze news seamlessly! (Ensure the FastAPI backend is running for it to work).

---

## 🧠 How the XAI Works
When text is submitted, the BiLSTM generates a confidence score for whether the text is Real or Fake. Simultaneously, the LIME explainer perturbs the text (removes words systematically) to see how the prediction changes. 
* Words highlighted in **Green** pushed the model towards "REAL".
* Words highlighted in **Red** pushed the model towards "FAKE".
* The darker the opacity, the stronger the influence of that word.

---
**Developed by the NewsGuard AI Team** 
*(Aditya Tiwari, Aman, Apoorv Chaturvedi, Anuj Pandey)*
