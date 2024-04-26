from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Vectorization parameters
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 2K features.
TOP_K = 2000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

def ngram_vectorize(train_texts, test_texts):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        test_texts: list, validation text strings.

    # Returns
        x_train, x_test: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'max_df': 0.9,
            'norm': 'l2', # Normalizes row vectors to have unit norm.
            'max_features': TOP_K,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary and idf from training texts and vectorize training texts (Return doc-term sparse matrix).
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_test = vectorizer.transform(test_texts)
    tokens = vectorizer.get_feature_names_out()
    print(f'\nTokens[:40]: {tokens[:40]}')
    print('\nNumber of tokens:', len(tokens))
    print(f'\nStop words: {vectorizer.get_stop_words()}')

    return x_train, x_test









