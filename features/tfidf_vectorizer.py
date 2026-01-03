from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path
MODEl_DIRS=Path('models')
MODEl_DIRS.mkdir(exist_ok=True)
VECTORIZER_PATH=MODEl_DIRS /'tfidf_vectorizer.pkl'


# Function to train tfidf:
def train_vectorizer(text):
    vectorizer=TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        stop_words='english'
    )
    X=vectorizer.fit_transform(text)
    joblib.dump(vectorizer,VECTORIZER_PATH)
    print(f"[INFO] the Tfidf vector save as {VECTORIZER_PATH}")
    return X
# function to load Tfidf:
def load_vectorizer():
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"vectorizer not found {VECTORIZER_PATH} train it first")
    vectorizer=joblib.load(VECTORIZER_PATH)
    return vectorizer