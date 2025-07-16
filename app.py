import streamlit as st
from predictor import predict_next_word


st.set_page_config(page_title="Next Word Predictor")
st.title("🔮 Next Word Predictor")
st.markdown("Enter a sentence with **at least 3 words**, and I will predict the next one!")

user_input = st.text_input("Your input")

if user_input:
    if len(user_input.split()) >= 3:
        prediction = predict_next_word(user_input)
        st.success(f"**Predicted next word:** `{prediction}`")
    else:
        st.warning("Please enter at least 3 words.")
