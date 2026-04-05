from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.preprocessing import tokenize_text


class TfidfLogisticRegression:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.model = None
        self.spam_threshold = 0
        self.c = 0

    def _vectorize_messages(self, messages, fit=False):
        if fit:
            return self.vectorizer.fit_transform(messages)
        return self.vectorizer.transform(messages)

    def train(self, df, c=1.0, spam_threshold=0.5):
        if c <= 0:
            raise ValueError('c must be greater than 0')
        if not 0.0 <= spam_threshold <= 1.0:
            raise ValueError('spam_threshold must be between 0 and 1')

        self.df = df
        self.c = c
        self.spam_threshold = spam_threshold
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_text,
            token_pattern=None,
            lowercase=False,
        )
        x_train = self._vectorize_messages(df['message'], fit=True)
        y_train = df['label']

        # L2-regularized logistic regression over TF-IDF features
        self.model = LogisticRegression(
            C=c,
            max_iter=1000,
            solver='liblinear',
            random_state=1729,
        )
        self.model.fit(x_train, y_train)

    def eval_message(self, message):
        if self.vectorizer is None or self.model is None:
            raise ValueError('Model is not trained. Call train() before eval_message().')

        tokens = tokenize_text(message)
        x_message = self._vectorize_messages([message])
        ham_prob, spam_prob = self.model.predict_proba(x_message)[0]
        logit_score = float(self.model.decision_function(x_message)[0])

        prediction = 'spam' if spam_prob >= self.spam_threshold else 'ham'

        details = {
            'tokens': tokens,
            'logit_score': logit_score,
            'nonzero_features': int(x_message.nnz),
        }
        return (prediction, (float(ham_prob), float(spam_prob)), details)
