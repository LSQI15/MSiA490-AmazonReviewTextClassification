from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from helper import model_training_testing
import pandas as pd


def train_test_logistic_regression_models(X_train, X_test, y_train, y_test):
    """
    function to train and test different logistic regression model
    :param X_train: X (features) in the training set
    :param X_test: X  (features) in the test set
    :param y_train: y (labels) in the training set
    :param y_test: y (labels) in the test set
    """
    # baseline model
    print('\nlogistic_regression_model0')
    model0 = LogisticRegression(random_state=115, solver='liblinear')
    model_training_testing(model0, X_train, X_test, y_train, y_test, (1, 1), 'Model_Results/LR_0')

    # model 1
    print('\nlogistic_regression_model1')
    model1 = LogisticRegression(random_state=115, solver='liblinear')
    model_training_testing(model1, X_train, X_test, y_train, y_test, (1, 2), 'Model_Results/LR_1')

    # model 2
    print('\nlogistic_regression_model2')
    model2 = LogisticRegression(random_state=115, solver='liblinear', penalty='l1')
    model_training_testing(model2, X_train, X_test, y_train, y_test, (1, 1), 'Model_Results/LR_2')

    # model 3
    print('\nlogistic_regression_model3')
    model3 = LogisticRegression(random_state=115, solver='liblinear', C=2)
    model_training_testing(model3, X_train, X_test, y_train, y_test, (1, 1), 'Model_Results/LR_3')


def main():
    """
    main function to train and test different logistic regression models
    :return: None
    """
    num_rows = 500000
    review_df = pd.read_csv("Data/processed_kindle_store_reviews.csv").dropna().head(num_rows)
    # train and test set split
    X_train, X_test, y_train, y_test = train_test_split(review_df['reviewText'], review_df['score'],
                                                        random_state=115)
    # call the function to train and test different models
    train_test_logistic_regression_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
