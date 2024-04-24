from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras import models, layers

import matplotlib.pyplot as plt

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 10

def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float64')
    x_val = selector.transform(x_val).astype('float64')
    return x_train, x_val

def plot_metrics(history):
    # Extract the training and validation metrics
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

    # Plot the loss curves
    axes[0].plot(train_loss, label='Training Loss')
    axes[0].plot(val_loss, label='Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()

    # Plot the accuracy curves
    axes[1].plot(train_mae, label='Training Mean Absolute Error')
    axes[1].plot(val_mae, label='Validation Mean Absolute Error')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Display the plot
    plt.show()



def create_regression_mlp(input_shape, x_train, train_labels, epochs, validation_data, batch_size):
    # Define the model architecture
    model = models.Sequential([
        layers.Dense(16, activation='relu', input_shape=input_shape),
        layers.Dense(1)  # Output layer with single unit for regression
    ])
    
    # Compile the model
    #opt = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    opt='adam'
    model.compile(optimizer=opt, loss='mean_squared_error')
    
    # Train the model
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        validation_data=validation_data,
        verbose=2,
        batch_size=batch_size
    )
    
    # Print results
    print('Validation loss:', history.history['val_loss'][-1])
    
    return model