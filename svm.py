from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from helper import model_training_testing
import pandas as pd


def train_test_svm_models(X_train, X_test, y_train, y_test):
    """
    function to train and test different logistic regression model
    :param X_train: X (features) in the training set
    :param X_test: X  (features) in the test set
    :param y_train: y (labels) in the training set
    :param y_test: y (labels) in the test set
    """
    # # baseline linear SVM model
    # print('\nlinearSVM_model0')
    # linearSVM_model0 = LinearSVC(random_state=115)
    # model_training_testing(linearSVM_model0, X_train, X_test, y_train, y_test, (1, 1), 'Model_Results/SVM_0')
    #
    # # linear SVM model 1
    # print('\nlinearSVM_model1')
    # linearSVM_model1 = LinearSVC(random_state=115)
    # model_training_testing(linearSVM_model1, X_train, X_test, y_train, y_test, (1, 2), 'Model_Results/SVM_1')
    #
    # # linear SVM model 2
    # print('\nlinearSVM_model2')
    # linearSVM_model2 = LinearSVC(random_state=115, dual=False)
    # model_training_testing(linearSVM_model2, X_train, X_test, y_train, y_test, (1, 1), 'Model_Results/SVM_2')
    #
    # # linear SVM model 3
    # print('\nlinearSVM_model3')
    # linearSVM_model3 = LinearSVC(random_state=115, loss='hinge')
    # model_training_testing(linearSVM_model3, X_train, X_test, y_train, y_test, (1, 1), 'Model_Results/SVM_3')

    # linear SVM model 4
    print('\nlinearSVM_model4')
    linearSVM_model1 = LinearSVC(random_state=115)
    model_training_testing(linearSVM_model1, X_train, X_test, y_train, y_test, (1, 3), 'Model_Results/SVM_4')


def main():
    """
    main function to train and test different logistic regression models
    :return: None
    """
    num_rows = 500000
    review_df = pd.read_csv("Data/processed_video_reviews.csv").dropna().head(num_rows)
    # train and test set split
    X_train, X_test, y_train, y_test = train_test_split(review_df['reviewText'], review_df['score'],
                                                        random_state=115)
    # call the function to train and test different models
    train_test_svm_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
