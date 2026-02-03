import pandas as pd
import re
import string
import pickle
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# ---------- NLTK ----------
nltk.download('stopwords')
nltk.download('wordnet')

# ---------- Ensure model folder ----------
os.makedirs("model", exist_ok=True)

# ---------- Load all datasets ----------
paths = [
    "data/reviews_badminton/data.csv",
    "data/reviews_tawa/data.csv",
    "data/reviews_tea/data.csv"
]

dfs = []
for p in paths:
    temp = pd.read_csv(p)
    temp.columns = temp.columns.str.strip().str.lower()
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

# ---------- Identify correct columns safely ----------
rating_col = [c for c in df.columns if 'rating' in c][0]
review_col = [c for c in df.columns if 'review' in c and 'title' not in c][0]

df = df[[rating_col, review_col]].dropna()
df.columns = ['rating', 'review']

# ---------- Label creation ----------
def label_sentiment(r):
    if r >= 4:
        return 1   # Positive
    elif r <= 2:
        return 0   # Negative
    else:
        return None  # Neutral (drop)

df['sentiment'] = df['rating'].apply(label_sentiment)
df = df.dropna()

# ---------- Text preprocessing ----------
lemmatizer = WordNetLemmatizer()

# 🚨 IMPORTANT: keep "not", "no", "never"
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['review'].apply(clean_text)

# ---------- TF-IDF (FINAL CONFIG) ----------
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),      # unigrams + bigrams
    min_df=3,
    sublinear_tf=True
)

X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment']

# ---------- Train/Test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------- Model ----------
model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    solver='liblinear'
)

# ---------- Train ----------
model.fit(X_train, y_train)

# ---------- Evaluation ----------
y_pred = model.predict(X_test)
print("\nF1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------- Save ----------
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\n✅ Model and vectorizer saved successfully")