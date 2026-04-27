import numpy as np
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .preprocessing import clean_text

class XAIExplainer:
    def __init__(self, model, tokenizer, max_length=50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        # class_names: index 0 -> Fake, index 1 -> Real
        self.explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

    def predict_proba(self, texts):
        # Clean texts and predict probabilities for both classes for LIME
        cleaned = [clean_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Model returns probability of being class 1 (REAL)
        preds = self.model.predict(padded, verbose=0)
        
        # We need to return an array of shape (len(texts), 2) for LIME [P(Fake), P(Real)]
        probas = np.zeros((len(texts), 2))
        probas[:, 1] = preds.flatten() # P(Real)
        probas[:, 0] = 1.0 - probas[:, 1] # P(Fake)
        return probas

    def explain(self, text):
        # Generate explanation for the given text
        # predict_proba method is passed to the explainer
        exp = self.explainer.explain_instance(
            text, 
            self.predict_proba, 
            num_features=10, 
            num_samples=100  # Smaller sample size for faster processing, acceptable for demo
        )
        return exp.as_list() # Returns list of tuples (word, weight)
