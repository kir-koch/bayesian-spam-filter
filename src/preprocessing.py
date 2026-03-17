import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

_STEMMER = PorterStemmer()
_STOP_WORDS = set(stopwords.words('english'))


def tokenize_text(text, stemmer=None, stop_words=None):
    stemmer = stemmer or _STEMMER
    stop_words = stop_words or _STOP_WORDS
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = [stemmer.stem(w)
              for w in text.split()
              if w not in stop_words and
              len(w) > 2]
    return tokens
