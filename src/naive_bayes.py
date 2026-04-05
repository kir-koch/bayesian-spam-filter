from collections import Counter, defaultdict
from math import log, exp
from src.preprocessing import tokenize_text


class NaiveBayes:
    def __init__(self):
        self.df = None
        self.lk_tokens_ham = None
        self.lk_tokens_spam = None
        self.prior_ham = 0
        self.prior_spam = 0
        self.spam_threshold = 0
        self.alpha = 0
        self.vocab_size = 0

    def _count_tokens(self, label):
        counter = Counter()
        series = self.df.query(f'label == {label}')['tokens']
        for words in series:
            counter.update(words)
        return counter, len(series)

    def _find_likelihoods_log(self, word_freq, alpha, vocab_size):
        total_tokens = sum(word_freq.values())
        denom = total_tokens + alpha * vocab_size

        # Laplace smoothing to avoid zero probabilities
        likelihoods = defaultdict(lambda: log(alpha / denom))
        for word, freq in word_freq.items():
            likelihoods[word] = log((freq + alpha) / denom)
        return likelihoods

    def train(self, df, alpha=1.0, spam_threshold=0.5):
        if alpha <= 0:
            raise ValueError('alpha must be greater than 0')
        if not 0.0 <= spam_threshold <= 1.0:
            raise ValueError('spam_threshold must be between 0 and 1')

        self.df = df
        self.alpha = alpha
        self.spam_threshold = spam_threshold
        token_freqs_ham, size_ham = self._count_tokens(0)
        token_freqs_spam, size_spam = self._count_tokens(1)
        self.vocab_size = len(set(token_freqs_ham) | set(token_freqs_spam))

        # Log-space for numerical stability
        self.lk_tokens_ham = self._find_likelihoods_log(
            token_freqs_ham,
            alpha,
            self.vocab_size,
        )
        self.lk_tokens_spam = self._find_likelihoods_log(
            token_freqs_spam,
            alpha,
            self.vocab_size,
        )

        self.prior_ham = log(size_ham / (size_ham + size_spam))
        self.prior_spam = log(size_spam / (size_ham + size_spam))

    def eval_message(self, message):
        if self.lk_tokens_ham is None or self.lk_tokens_spam is None:
            raise ValueError('Model is not trained. Call train() before eval_message().')

        tokens = tokenize_text(message)

        likelihood_ham = sum(self.lk_tokens_ham[w] for w in tokens)
        likelihood_spam = sum(self.lk_tokens_spam[w] for w in tokens)

        numer_ham_log = self.prior_ham + likelihood_ham
        numer_spam_log = self.prior_spam + likelihood_spam

        # Stable softmax to avoid overflow on long messages
        max_log = max(numer_ham_log, numer_spam_log)
        ham_weight = exp(numer_ham_log - max_log)
        spam_weight = exp(numer_spam_log - max_log)
        normalizer = ham_weight + spam_weight

        postr_ham = ham_weight / normalizer
        postr_spam = spam_weight / normalizer

        prediction = 'spam' if postr_spam >= self.spam_threshold else 'ham'

        details = {
            'tokens': tokens,
            'log_joint_ham': numer_ham_log,
            'log_joint_spam': numer_spam_log,
        }
        return (prediction, (postr_ham, postr_spam), details)
