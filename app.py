import streamlit as st
import csv
from keras.models import load_model
import numpy as np
import tensorflow as tf

# Load the end-to-end model (includes TextVectorization layer)
model = load_model("best_lstm_textvec.keras")

# Logging function remains the same
def log_user_input(review, prediction):
    with open("user_feedback.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([review, prediction])

# Streamlit UI
st.title("🎬 Sentiment Analysis App")
st.write("Type a movie review below and see the sentiment:")

user_input = st.text_area("Your Review", height=150)

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Directly pass the raw string into the model
        input_tensor = tf.constant([user_input])
        pred_array = model.predict(input_tensor)
        pred = float(pred_array[0][0])
        sentiment = "Positive 😊" if pred >= 0.5 else "Negative 😞"
        st.subheader(sentiment)
        st.write(f"Confidence: {pred:.2f}")
        st.write(f"{pred_array}")

        # Feedback logging
        user_label = st.radio("Was this prediction correct?", ("Yes", "No"))
        if user_label == "No":
            true_label = 1 - int(pred >= 0.5)
            log_user_input(user_input, true_label)
        else:
            log_user_input(user_input, int(pred >= 0.5))
