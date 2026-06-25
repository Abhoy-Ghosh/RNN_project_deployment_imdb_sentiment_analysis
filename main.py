# ============================================================
# IMDB Movie Review Sentiment Analysis using Simple RNN
# Streamlit Application
# ============================================================

import re
import streamlit as st
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# ------------------------------------------------------------
# Load Dataset Vocabulary
# ------------------------------------------------------------
word_index = imdb.get_word_index()

# ------------------------------------------------------------
# Load Trained Model
# ------------------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    return load_model("notebook\simple_rnn_imdb.h5")

model = load_sentiment_model()

# ------------------------------------------------------------
# Text Preprocessing
# ------------------------------------------------------------
def preprocess_text(text):

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Lowercase
    text = text.lower()

    # Tokenize
    words = text.split()

    # Convert words into integers
    encoded = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    # Pad sequence
    padded = sequence.pad_sequences(
        [encoded],
        maxlen=400,
        padding="pre",
        truncating="pre"
    )

    return padded


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below. The trained **Simple RNN model** "
    "will predict whether the review is **Positive** or **Negative**."
)

st.divider()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:

    st.header("📖 About")

    st.write("""
This application uses:

- TensorFlow / Keras
- Simple RNN
- IMDB Dataset
- Binary Sentiment Classification
""")

    st.divider()

    st.subheader("💬 Example Reviews")

    if st.button("😊 Positive Example"):
        st.session_state.review = (
            "This movie was absolutely fantastic. "
            "The acting was brilliant and I loved every minute of it."
        )

    if st.button("☹️ Negative Example"):
        st.session_state.review = (
            "The movie was boring, slow, predictable and a complete waste of time."
        )

# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------
if "review" not in st.session_state:
    st.session_state.review = ""

# ------------------------------------------------------------
# User Input
# ------------------------------------------------------------
review = st.text_area(
    "✍️ Movie Review",
    value=st.session_state.review,
    height=180,
    placeholder="Type your movie review here..."
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if st.button("🚀 Analyze Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
        st.stop()

    processed = preprocess_text(review)

    with st.spinner("Analyzing Review..."):

        prediction = model.predict(processed, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability
    else:
        sentiment = "Negative"
        confidence = 1 - probability

    st.divider()

    st.subheader("Prediction Result")

    if sentiment == "Positive":
        st.success("😊 Positive Review")
    else:
        st.error("☹️ Negative Review")

    st.metric(
        label="Confidence",
        value=f"{confidence*100:.2f}%"
    )

    st.progress(confidence)

    st.divider()

    st.subheader("📄 Review")

    st.info(review)

    st.divider()

    if sentiment == "Positive":
        st.write(
            "The model predicts that this review expresses an **overall positive opinion**."
        )
    else:
        st.write(
            "The model predicts that this review expresses an **overall negative opinion**."
        )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()

st.caption(
    "Built with ❤️ using Streamlit, TensorFlow, and Keras"
)