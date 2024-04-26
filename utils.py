from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
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
    print(tokens[:40])
    print('Number of tokens:', len(tokens))

    return x_train, x_test







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