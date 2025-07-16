# Next Word Prediction using LSTM

This project is a deep learning-based text prediction system that suggests the next word in a sentence using a trained LSTM model. It is trained on Wikipedia data and provides predictions through both command-line and web interfaces.

## Features

- Predicts the next word based on a sequence of input words
- Uses LSTM (Long Short-Term Memory) neural networks
- Trained on real-world text from Wikipedia
- Web-based interface built using Streamlit
- Easy to retrain with new data or topics

## Project Structure

next_word_prediction_project/
├── app.py # Streamlit web app for prediction
├── train_model.py # Script to train the LSTM model
├── predictor.py # CLI-based prediction script
├── fetch_wikipedia_data.py # Script to fetch and save Wikipedia articles
├── utils.py # Utility functions for preprocessing
├── data/
│ └── raw_text.txt # Text corpus collected from Wikipedia
├── model/
│ └── model.h5 # Trained LSTM model
└── tokenizer/
└── tokenizer.pkl # Tokenizer object used for encoding words

