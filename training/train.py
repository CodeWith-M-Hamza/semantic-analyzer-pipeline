import joblib 
from pathlib import Path
from sklearn.svm import LinearSVC
from training.data_loader import load_raw_data
from features.tfidf_vectorizer import load_vectorizer

from preprocessing.text_cleaner import clean_text


MODELS_DIR=Path('models')
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH=MODELS_DIR/'sentiment_model.pkl'


df=load_raw_data()
df['clean_text']=df['Text'].apply(clean_text)


vectoizer=load_vectorizer()
X_train=vectoizer.transform(df['clean_text'])


y_train=df['Sentiments']
model=LinearSVC(max_iter=2000)
model.fit(
    X_train,
    y_train
)


joblib.dump(model,MODEL_PATH)
print(f"trained model saved at {MODEL_PATH}")