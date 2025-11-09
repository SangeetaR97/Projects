# mental_health_app.py

import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("mental_health_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))  # Save separately during training

# Text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

# Recommendation system
def provide_recommendation(label):
    recommendations = {
        "Normal": "✅ You seem to be in a balanced state. Keep practicing healthy routines and mindfulness.",
        "Depression": "😔 You might be feeling low. Please talk to someone you trust or consider professional support.",
        "Anxiety": "😟 If you’re feeling anxious, try breathing exercises or meditation. Seek help if it persists.",
        "Suicidal": "🚨 Please seek immediate help from a mental health professional or helpline.",
        "Personality disorder": "🧠 Therapy can help manage experiences related to personality challenges.",
        "Other": "📘 Monitor your state and journal thoughts. Seek help if needed."
    }
    return recommendations.get(label, "We're here for you. Consider reaching out for support.")

# Streamlit UI
st.set_page_config(page_title="Mental Health Classifier", layout="centered")

st.title("🧠 Mental Health Analysis")
st.markdown("Enter a short text describing your current feelings.")

user_input = st.text_area("Enter your statement here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid input.")
    else:
        cleaned_input = clean_text(user_input)
        transformed_input = tfidf.transform([cleaned_input])
        prediction = model.predict(transformed_input)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.subheader(f"Predicted Status: `{predicted_label}`")
        st.markdown(provide_recommendation(predicted_label))
