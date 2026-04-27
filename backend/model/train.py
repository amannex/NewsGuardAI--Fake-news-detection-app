import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
import pickle
import os

# Set paths for the real ISOT dataset
base_dir = os.path.dirname(os.path.dirname(__file__))
fake_csv_path = os.path.join(base_dir, 'data', 'Fake.csv')
true_csv_path = os.path.join(base_dir, 'data', 'True.csv')

if os.path.exists(fake_csv_path) and os.path.exists(true_csv_path):
    print("--- Loading Large Dataset (ISOT) ---")
    fake_df = pd.read_csv(fake_csv_path)
    true_df = pd.read_csv(true_csv_path)
    
    # Create labels: 0 = FAKE, 1 = REAL
    fake_df['label'] = 0
    true_df['label'] = 1
    
    # Combine and shuffle
    df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Optional: For speed during testing, you can limit the dataset size here
    # df = df.head(50000) 
    
    texts = df['text'].astype(str).tolist()
    labels = df['label'].values
    
    # Larger parameters for a real dataset
    vocab_size = 10000
    max_length = 200
else:
    print("--- Large dataset not found. Using Dummy Dataset ---")
    texts = [
        "The president just announced a brand new policy that will impact the economy heavily, reliable experts say.", # REAL
        "Aliens discovered building pyramids on Mars base! Shocking video evidence that the government is hiding.", # FAKE
        "Stock market sees a slight bump today after tech companies release their quarterly earnings reports.", # REAL
        "You won't believe this miracle cure! Doctors hate it. Drink this one liquid to instantly lose 50 pounds.", # FAKE
        "Local school district votes to increase funding for the arts program next year.", # REAL
        "Breaking: Famous celebrity found alive in secret underground bunker after supposedly passing away 10 years ago.", # FAKE
        "The new smartphone from TechCorp features a better battery and upgraded camera system.", # REAL
        "Earth is actually flat, say scientists who were recently silenced by the deep state.", # FAKE
        "City council approves road repair budget for the upcoming fiscal quarter.", # REAL
        "Zombie virus outbreak reported in a remote village! CDC covering it up!", # FAKE
    ] * 10  # Duplicate to have enough samples for training

    # 0 = FAKE, 1 = REAL
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10
    labels = np.array(labels)

    # Smaller parameters for dummy
    vocab_size = 1000
    max_length = 50
embedding_dim = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

print("--- Tokenizing Text ---")
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print("--- Building the BiLSTM Model ---")
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("--- Training the Model ---")
# Quickly train on the dummy dataset
model.fit(padded, labels, epochs=5, verbose=1)

print("--- Saving Model and Tokenizer ---")
os.makedirs(os.path.dirname(__file__), exist_ok=True)
model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer.pkl')

model.save(model_path)
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"Saved artifacts to {model_path} and {tokenizer_path}")
