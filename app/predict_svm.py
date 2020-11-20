import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def predict_ratings(review_to_predict, best_svm, tfidf_vectorizer):
    """
    function to make predictions for reviews based using the best svm model
    :param review_to_predict: a list of reviews
    :param best_svm: sklearn model object for the best svm model
    :param tfidf_vectorizer: tfidf_vectorizer to transform raw text to tfidf representation
    :return: a json object containing reviews and their corresponding predicted ratings
    """
    tfidf_review = tfidf_vectorizer.transform(review_to_predict)
    predicted_rating = best_svm.predict(tfidf_review)
    combined_df = pd.DataFrame(list(zip(review_to_predict, predicted_rating)),
                               columns=['review_to_predict', 'predicted_rating'])
    return combined_df.to_json(orient='index')


def main():
    """
    main function to used saved svm model to make review predictions
    :return: None
    """
    # load saved artifacts to make the prediction
    with open('best_svm.pkl', 'rb') as f:
        best_svm = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    # prediction example
    review_to_predict \
        = ['this movie is really interesting. i like it',
           'the plot is super boring and i fell asleep during the movie.',
           'Perfectly serviceable and utterly forgettable, Honest Thief nonetheless offers a few pleasing details to keep it from being a total slog.',
           'On the surface, this Ferzan Ozpetek film might seem to be a classic melodrama, but its details are entirely contemporary.',
           'a bizarre psychological thriller about the real effects of trauma.',
           'Bill Murray scores again with his signature dry humor.',
           'It\'s a percussionist\'s paradise which demonstrates a perfect rhythm between effervescent joy and no-nonsense political urgency.'
           ]
    predictions = predict_ratings(review_to_predict, best_svm, tfidf_vectorizer)
    print(predictions)


if __name__ == "__main__":
    main()