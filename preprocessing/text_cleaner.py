import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
stop_words=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()
def clean_text(text:str)-> str:
    # Lower-case
    text=text.lower()
    # Remove urls and mentioned hashed tags:
    text=re.sub(r'http\S+|www\S+','',text)
    text=re.sub(r'@\w+|#\w+','',text)
    # Remove Punctuation:
    text=re.sub(r'[^\w\s]','',text)
    # Tokenize and remove stopwords
    words=[lemmatizer.lemmatize(words) for words in text.split() if words not in stop_words]  
    return ' '.join(words)
 