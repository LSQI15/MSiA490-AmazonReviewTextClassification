import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import argparse
import sys
import numpy as np


def main(NUM_EPOCHS, BATCH_SIZE, MAX_SEQUENCE_LENGTH, model_name):
    """
    Main function to train and test LSTM model
    :param NUM_EPOCHS: number of epoch to train the model
    :param BATCH_SIZE: batch size for training the model
    :param MAX_SEQUENCE_LENGTH: max length of input review string
    :param model_name: name of the LSTM model
    :return: None
    """
    # read in data
    df = pd.read_csv("s3://msia490project/processed_video_reviews.csv").head(500000)
    df['reviewText'] = df['reviewText'].astype(str)

    # Hyper-parameter for the model
    MAX_NB_WORDS = 50000
    EMBEDDING_DIM = 100
    MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
    NUM_EPOCHS = NUM_EPOCHS
    BATCH_SIZE = BATCH_SIZE

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['reviewText'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df['reviewText'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)

    y = pd.get_dummies(df['score']).values
    print('Shape of label tensor:', y.shape)

    # Training set and test set split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Model Structure
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Train the LSTM  Model
    print('Start training BERT model.')
    print('Number of epochs: ', NUM_EPOCHS)
    print('Max input length: ', MAX_SEQUENCE_LENGTH)
    print('Batch size: ', BATCH_SIZE)
    history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)])

    # Evaluate the model on the test set
    accuracy = model.evaluate(X_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))

    # Confusion Matrix
    pred = model.predict(X_test)
    labels = ['1', '2', '3', '4', '5']
    y_pred_labels = [labels[x] for x in [np.argmax(x) for x in pred]]
    y_test_labels = [labels[x] for x in [np.argmax(x) for x in y_test]]
    confusion = confusion_matrix(y_test_labels, y_pred_labels)
    confusion_df = pd.DataFrame(confusion,index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])
    print(confusion_df)
    sns.heatmap(confusion_df/np.sum(confusion_df), annot=True, fmt='.2%', cmap='Blues')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig(model_name + '-confusion_matrix' + '.png')
    plt.clf()

    # Visualize training loss and validation accuracy
    plt.title('Training Loss vs Validation Accuracy')
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_accuracy'], label='test_accuracy')
    plt.legend()
    plt.savefig(model_name + '.png')
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", help="number of epochs for training the model")
    parser.add_argument("--batch_size", help="batch_size for training the model")
    parser.add_argument("--max_length", help="max length of reviews")
    args = parser.parse_args()
    model_name = 'LSTM-max_length' + args.max_length + 'batch_size' + args.batch_size + 'num_epoch' + args.num_epoch
    sys.stdout = open(model_name + '.txt', 'w')
    main(int(args.num_epoch), int(args.batch_size), int(args.max_length), model_name)
