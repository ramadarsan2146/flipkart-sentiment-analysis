import streamlit as st
import pickle
import re
import string
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- NLTK setup (download once) ----------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ---------- Load model safely ----------
@st.cache_resource
def load_models():
    with open("model/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_models()

lemmatizer = WordNetLemmatizer()

# ---------- Stopwords (keep negations) ----------
stop_words = set(stopwords.words("english"))
NEGATION_WORDS = {"not", "no", "never", "hardly", "n't"}
stop_words = stop_words - NEGATION_WORDS

# ---------- Strong negative phrases ----------
NEGATIVE_PHRASES = [
    "not good",
    "not worth",
    "not satisfied",
    "waste of money",
    "very bad",
    "worst",
    "poor quality",
    "bad quality"
]

# ---------- Text Cleaning ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------- UI ----------
st.set_page_config(page_title="Flipkart Sentiment Analysis")
st.title("🛒 Flipkart Product Review Sentiment Analysis")
st.write("Analyze whether a Flipkart product review is **Positive** or **Negative**")

review = st.text_area("Enter a product review:")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)

        # 🚨 Rule-based NEGATIVE override (on cleaned text)
        if any(phrase in cleaned for phrase in NEGATIVE_PHRASES):
            st.error("❌ Negative Sentiment")
            st.write("Confidence (Positive): 0.00")
            st.caption("Detected strong negative phrase")

        else:
            vector = tfidf.transform([cleaned])
            probs = model.predict_proba(vector)[0]
            classes = model.classes_

            # Safely get positive probability
            positive_index = list(classes).index(1)
            positive_prob = probs[positive_index]

            # ✅ Correct threshold
            if positive_prob >= 0.50:
                st.success("✅ Positive Sentiment")
            else:
                st.error("❌ Negative Sentiment")

            st.write(f"Confidence (Positive): {positive_prob:.2f}")