from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from helper import model_training_testing
import pandas as pd


def train_test_MultinomialNB_models(X_train, X_test, y_train, y_test):
    """
    function to train and test different MultinomialNB models
    :param X_train: X (features) in the training set
    :param X_test: X  (features) in the test set
    :param y_train: y (labels) in the training set
    :param y_test: y (labels) in the test set
    """
    # baseline MultinomialNB model
    print('\nMultinomialNB_model0')
    MultinomialNB_model0 = MultinomialNB(alpha=1.0, fit_prior=True)
    model_training_testing(MultinomialNB_model0, X_train, X_test, y_train, y_test, (1, 1), 'Model_Results/MultinomialNB_0')

    # MultinomialNB model 1
    print('\nMultinomialNB_model1')
    MultinomialNB_model1 = MultinomialNB(alpha=0, fit_prior=True)
    model_training_testing(MultinomialNB_model1, X_train, X_test, y_train, y_test, (1, 2), 'Model_Results/MultinomialNB_1')

    # MultinomialNB model 2
    print('\nMultinomialNB_model2')
    MultinomialNB_model2 = MultinomialNB(alpha=0, fit_prior=True)
    model_training_testing(MultinomialNB_model2, X_train, X_test, y_train, y_test, (1, 1), 'Model_Results/MultinomialNB_2')

    # MultinomialNB 3
    print('\nMultinomialNB_model3')
    MultinomialNB_model3 = MultinomialNB(alpha=0, fit_prior=False)
    model_training_testing(MultinomialNB_model3, X_train, X_test, y_train, y_test, (1, 2), 'Model_Results/MultinomialNB_3')


def main():
    """
    main function to train and test different MultinomialNB models
    :return: None
    """
    num_rows = 500000
    review_df = pd.read_csv("Data/processed_video_reviews.csv").dropna().head(num_rows)
    # train and test set split
    X_train, X_test, y_train, y_test = train_test_split(review_df['reviewText'], review_df['score'],
                                                        random_state=115)
    # call the function to train and test different models
    train_test_MultinomialNB_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()