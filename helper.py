from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score
import csv


def tfidf_vectorizer(X_train, X_test, ngram_range):
    """
    function to remove stopwords and transform raw text to tfidf representation
    :param X_train: X (features) in the training set
    :param X_test: X  (features) in the test set
    :param ngram_range: the lower and upper boundary of the range of n-values for different word n-grams to be extracted
    :return: X_train_tfidf, X_test_tfidf, which are tfidf representations of X_train and X_test
    """
    # initialize  the tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=ngram_range, stop_words='english')
    # fit it to X_train
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(X_train)
    # apply it to X_trian and X_test
    X_train_tfidf = tfidf_vectorizer.transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    # return the transformed X_train and X_test
    return X_train_tfidf, X_test_tfidf


def model_training_testing(model, X_train, X_test, y_train, y_test, ngram_range, out_file_path):
    """
    function to train a given model and test its performance in the test set
    :param model: model to train and test
    :param X_train: X (features) in the training set
    :param X_test: X  (features) in the test set
    :param y_train: y (labels) in the training set
    :param y_test: y (labels) in the test set
    :param ngram_range: the lower and upper boundary of the range of n-values for different word n-grams to be extracted
    :param out_file_path: path to txt file that stores model results
    """
    # transform input data using tfidf
    X_train_tfidf, X_test_tfidf = tfidf_vectorizer(X_train, X_test, ngram_range)
    # train the model
    clf = model.fit(X_train_tfidf, y_train)
    # evaluate model performance in the test set
    y_pred = clf.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    accuracy = model.score(X_test_tfidf, y_test)
    precision, recall, fscore, train_support = score(y_test, y_pred, average='weighted')

    # output model results to a txt file
    outF = open(out_file_path, "w")
    outF.write('Model Parameter: ')
    outF.write(str(model).replace('\n                   ', ''))
    outF.write('\nBOW representation: TFIDF')
    outF.write('\nAccuracy: ' + str(accuracy))
    outF.write('\nPrecision: ' + str(precision))
    outF.write('\nRecall: ' + str(recall))
    outF.write('\nFscore: ' + str(fscore))
    outF.write('\nClassification Report\n')
    outF.writelines(report_df.to_string())
    outF.close()
