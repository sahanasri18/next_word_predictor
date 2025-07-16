import re
import nltk
from nltk.tokenize import word_tokenize
import pickle

nltk.download("punkt")
nltk.download("punkt_tab")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_corpus(text):
    return word_tokenize(clean_text(text))

def save_tokenizer(tokenizer):
    with open("model/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

def load_tokenizer():
    with open("model/tokenizer.pkl", "rb") as f:
        return pickle.load(f)
