import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- NLTK ----------
nltk.download('stopwords')
nltk.download('wordnet')

# ---------- Load model ----------
model = pickle.load(open("model/sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()

# IMPORTANT: must match training
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

# Minimal strong negative override (realistic & acceptable)
NEGATIVE_KEYWORDS = [
    "waste", "worst", "useless", "broken",
    "damaged", "fake", "pathetic", "cheap quality"
]

# ---------- Text Cleaning ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
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

        # Rule-based override for extreme negatives
        if any(word in cleaned for word in NEGATIVE_KEYWORDS):
            st.error("❌ Negative Sentiment")
            st.write("Confidence (Positive): 0.00")
            st.caption("Detected strong negative keywords")
        else:
            vector = tfidf.transform([cleaned])

            # ✅ FINAL decision from model.predict()
            prediction = model.predict(vector)[0]

            # Probabilities only for display
            probs = model.predict_proba(vector)[0]
            prob_map = dict(zip(model.classes_, probs))
            positive_prob = prob_map.get(1, 0.0)

            if prediction == 1:
                st.success("✅ Positive Sentiment")
            else:
                st.error("❌ Negative Sentiment")

            st.write(f"Confidence (Positive): {positive_prob:.2f}")