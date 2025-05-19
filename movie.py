import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Inject CSS for background image and styled container

# Load model and tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('gru_model.keras')

@st.cache_data
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

MAX_SEQUENCE_LENGTH = 100

# Begin form
with st.form("titanic_form"):
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.write("Enter a movie review below to predict its sentiment.")

    review = st.text_area("Movie Review", height=150)

    # ✅ Submit button required for forms
    submit = st.form_submit_button("Predict Sentiment")

    if submit:
        if review:
            sequence = tokenizer.texts_to_sequences([review])
            padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
            prediction = model.predict(padded)[0][0]
            sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
            st.markdown(f"**Sentiment:** {sentiment}")
        else:
            st.warning("Please enter a movie review to analyze.")
