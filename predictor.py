import numpy as np
from keras.models import load_model
from utils import clean_text, tokenize_corpus, load_tokenizer
from nltk.tokenize import word_tokenize

model = load_model("model/model.h5")
tokenizer = load_tokenizer()
seq_length = 3  # based on training

def predict_next_word(input_text, top_n=3):
    cleaned = clean_text(input_text)
    tokens = word_tokenize(cleaned)
    if len(tokens) < seq_length:
        return ["⚠️ Not enough words. Enter at least 3."]
    
    input_seq = tokens[-seq_length:]
    input_encoded = tokenizer.texts_to_sequences([input_seq])[0]

    if len(input_encoded) < seq_length:
        input_encoded = [0] * (seq_length - len(input_encoded)) + input_encoded

    input_array = np.array(input_encoded).reshape(1, seq_length)

    predictions = model.predict(input_array, verbose=0)[0]
    top_indices = predictions.argsort()[-top_n:][::-1]
    words = [word for word, index in tokenizer.word_index.items() if index in top_indices]

    return words

