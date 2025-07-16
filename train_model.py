import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import tokenize_corpus, save_tokenizer
import os

# Load and preprocess text
with open('data/raw_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokens = tokenize_corpus(text)

# Check token length
if len(tokens) < 5:
    raise ValueError("The text is too short. Please add more content to data/raw_text.txt.")

# Create input-output sequences
window = 3
input_sequences = []
targets = []

for i in range(window, len(tokens)):
    input_sequences.append(tokens[i-window:i])
    targets.append(tokens[i])

print("Sample input sequences:", input_sequences[:2])

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens)
word_index = tokenizer.word_index

# Convert to numeric
X = []
y = []

for seq, label in zip(input_sequences, targets):
    encoded_seq = [word_index.get(w, 0) for w in seq]
    encoded_label = word_index.get(label, 0)
    X.append(encoded_seq)
    y.append(encoded_label)

X = np.array(X)
y = np.array(y)

# Reshape for LSTM: (samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Model
model = Sequential([
    Embedding(input_dim=len(word_index)+1, output_dim=50),
    LSTM(100),
    Dense(len(word_index)+1, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=30, batch_size=32)

# Save model and tokenizer
os.makedirs("model", exist_ok=True)
model.save('model/model.h5')
save_tokenizer(tokenizer)

print("✅ Training complete and model saved.")
