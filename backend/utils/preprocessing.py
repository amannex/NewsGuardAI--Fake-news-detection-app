import re

def clean_text(text: str) -> str:
    # Basic cleaning
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Removing extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
