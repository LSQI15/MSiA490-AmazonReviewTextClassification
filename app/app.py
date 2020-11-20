import logging.config
import pandas as pd
import pickle
import traceback
from flask import render_template, request, redirect, url_for
import logging.config
from flask import Flask
import logging
logging.getLogger().setLevel(logging.INFO)

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, template_folder="templates")
# load saved artifacts to make the prediction
with open('app/best_svm.pkl', 'rb') as f:
    best_svm = pickle.load(f)
with open('app/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():

    input_review = request.form["review"]
    logging.info(input_review)
    tfidf_review = tfidf_vectorizer.transform([input_review])
    predicted_rating = best_svm.predict(tfidf_review)
    logging.info(predicted_rating)

    return render_template('main.html',
                           prediction_text=str(predicted_rating[0]),
                           review=input_review)

if __name__ == '__main__':
    app.run()