import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import os


def generate_artifacts_for_best_svm_model(best_svm, ngram_range, X_train, y_train):
    """
    function to generate artifacts for the best svm model and save them locally
    :param best_svm: sklearn model object for the best svm model
    :param ngram_range: the lower and upper boundary of the range of n-values for different word n-grams to be extracted
    :param X_train: X (features) in the training set
    :param y_train: y (labels) in the training set
    :return: None
    """
    # transform input data using tfidf and save the tfidf_vectorizer locally
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=ngram_range)
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(X_train)
    X_train_tfidf = tfidf_vectorizer.transform(X_train)
    with open(os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    # train and save the model locally
    best_svm_model = best_svm.fit(X_train_tfidf, y_train)
    with open(os.path.join(os.path.dirname(__file__), 'best_svm.pkl'), 'wb') as f:
        pickle.dump(best_svm_model, f)


def main():
    """
    main function to generate artifacts for the best svm model
    :return: None
    """
    num_rows = 500000
    review_df = pd.read_csv("s3://msia490project/processed_video_reviews.csv").dropna().head(num_rows)
    # train and test set split
    X_train, X_test, y_train, y_test = train_test_split(review_df['reviewText'], review_df['score'],
                                                        random_state=115)
    # re-run the model pipeline and generate necessary artifacts for making predictions
    best_svm = LinearSVC(random_state=115)
    ngram_range = (1, 3)
    generate_artifacts_for_best_svm_model(best_svm, ngram_range, X_train, y_train)


if __name__ == "__main__":
    main()