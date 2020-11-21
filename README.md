### Northwestern University
### MSiA-490 Fall 2020
### Amazon Kindle Store Review Text Classification Project
### Siqi Li

<!-- toc -->

## Repo structure

<!-- toc -->

- [Project Topic](#project-topic)
- [Dataset](#dataset)
- [Web App for Amazon Review Text Classification](#web-app-for-amazon-review-text-classification)
- [Model Results](#model-results)
    * [1. Logistic Regression](#logistic-regression)
    * [2. Support Vector Machine](#support-vector-machine)
    * [3. Multinomial Naive Bayes](#multinomial-naive-bayes)
    * [4. LSTM](#lstm)
    * [5. Bidirectional LSTM](#bidirectional-lstm)
    * [6. Bidirectional LSTM with Self-Attention](#bidirectional-lstm-with-self-attention)
    * [7. BERT](#bert)
- [Citation](#citation)

<!-- toc -->

### Project Topic
In this project, I trained, fine-tuned, and evaluated 7 multi-class text classification approaches (logistic regression, 
support vector machine (SVM), multinomial naive bayes, BERT, LSTM, bidirectional LSTM, and bidirectional LSTM with 
self-attention). Specifically, each model was trained on the training set (80% of data) and evaluated on the test 
set (20% of data) using metrics such as accuracy, precision, recall, and F-1 score. The best model in terms of not only 
accuracy but also ease to use was also productized as a flask web app to take review text as input and the output the 
corresponding predicted review score (1-5).

### Dataset
The dataset chosen for this project is Amazon video review data from [Amazon Review Data](https://snap.stanford.edu/data/web-Amazon.html). 
The raw dataset contains information such as reviews (ratings, text, helpfulness votes) and product metadata 
(descriptions, category information, price, brand, and image features) for 717,651 reviews. Due to computational 
limitation, only the first 500,000 records are used in this text classification project. Each record has the overall 
review score (integer from 1 to 5) and the review text (string).

### Web App for Amazon Review Text Classification

To run the web app, please clone the Github repo and run the following command in terminal.

```shell script
git clone git@github.com:LSQI15/MSiA490-AmazonReviewTextClassification.git
cd MSiA490-AmazonReviewTextClassification
pip install -r requirements.txt
python3 app/generate_artifacts.py
phthon3 app/app.py
```

The app will then be accessible on http://127.0.0.1:5000/. Press  CTRL+C to at any time to quit.

<img src="https://github.com/LSQI15/MSiA490-AmazonReviewTextClassification/blob/main/app/static/demo.png" width="800">
    
### Model Results

* Among traditional machine learning models, support vector machine model out-performed logistic regression model and 
multinomial naïve bayes models. The best SVM model used L-2 penalty, square-hinge loss, TF-IDF feature representation 
of unigram, bigram, and trigram, and achieved a multi-class accuracy of 0.7284 on the test set. In addition, increasing 
n-gram range of TF-IDF from using only 1-gram to using 1-gram to 3-gram significantly helped boost the performance for 
logistic regression model (0.6548 to 0.6827) and SVM models (0.6659 to 0.7284).

* Various LSTM models were trained and evaluated, but they all have similar results (~69%). One thing I noticed was that 
batch size seemed to be related to model accuracy. Specifically, a smaller batch size seemed to lead to a higher accuracy
for bidirectional LSTM model. This reflects Kevin’s (2018) finding that higher batch sizes leaded to lower asymptotic 
test accuracy with a multi-layer perceptron (MLP) model. Moreover, adding a self-attention layer didn’t improve model 
performance a lot in this work. Other model parameters or structure are needed to be fine-tuned in order to boost LSTM’s
performance. Overall, the bidirectional LSTM model with an embedding size of 256, a max number of words of 100,000, a 
batch size of 64 and a self-attention layer had the best accuracy of 0.7091 on the test set.

* For BERT model, due to computational limits, only max input length was fine-tuned. The results show that the greater the
input length, the higher the classification accuracy. This could be explained by the fact that the average review 
length is 775.5, and with longer input length, BERT models can extract more information to make a better classification. 
The best BERT model had max length of input equal to 512 and a batch size equal to 8. It achieved an accuracy of 0.75802 
in the test set, which was an 3.91% increase from the best. SVM model. The following graph shows the normalized confusion 
matrix for the best BERT model. While the overall accuracy is satisfying, the model still has a potential for 
improvement, especially for correctly distinguishing. reviews with 4 and 5 stars.

<img src="https://github.com/LSQI15/MSiA490-AmazonReviewTextClassification/blob/main/Model_Results/BERT-max_length512batch_size8num_epoch7.png" width="800">

* One limitation is that while BERT model has the best text classification performance, it took a much longer time to 
train, compared to LSTM and other traditional machine learning models. The best BERT model, for instance, needed more 
than 5 hours to train an epoch using a NVIDIA GeForce RTX 2080 Ti GPU, while the best SVM model took only 20 minutes 
to train on an Intel 4-Core i7 CPU. If hardware permits, training BERT model with a higher max input length may lead 
to an even better performance than I currently had in this work.

#### Logistic Regression

| Logistic Regression Model   | TFIDF n-gram Range | Model Parameters  | Accuracy  | Precision  | Recall | F-score|
| :-----: | :----: | :----------:      | :-------: | :--------: | :----: | :----: |
| #0 | (1,1) | C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, l1_ratio=None, max_iter=100,multi_class='auto', n_jobs=None, penalty='l2',random_state=115, solver='liblinear', tol=0.0001, verbose=0,warm_start=False | 0.6545 | 0.6171 | 0.6545 | 0.6170 |
| #1 | (1,2) | C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, l1_ratio=None, max_iter=100,multi_class='auto', n_jobs=None, penalty='l2',random_state=115, solver='liblinear', tol=0.0001, verbose=0,warm_start=False | 0.6827 | 0.6580 | 0.6827 | 0.6513 |
| #2 | (1,1) | C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, l1_ratio=None, max_iter=100,multi_class='auto', n_jobs=None, penalty='l1',random_state=115, solver='liblinear', tol=0.0001, verbose=0,warm_start=False | 0.6491 | 0.6087 | 0.6491 | 0.6109 |
| #3 | (1,1) | C=2, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, l1_ratio=None, max_iter=100,multi_class='auto', n_jobs=None, penalty='l2',random_state=115, solver='liblinear', tol=0.0001, verbose=0,warm_start=False | 0.6591 | 0.6242| 0.6591 | 0.6263 |

#### Support Vector Machine

| Support Vector Machine Model   | TFIDF n-gram Range | Model Parameters  | Accuracy  | Precision  | Recall | F-score|
| :-----: | :----: | :----------:      | :-------: | :--------: | :----: | :----: |
| #0 | (1,1) | C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=115, tol=0.0001, verbose=0 | 0.6659 | 0.6351 | 0.6659 | 0.6396 |
| #1 | (1,2) | C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=115, tol=0.0001, verbose=0 | 0.7227 | 0.7061 | 0.7227 | 0.7071 |
| #2 | (1,1) | C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=115, tol=0.0001, verbose=0 | 0.6659 | 0.6351 | 0.6659 | 0.6396 | 
| #3 | (1,1) | C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=115, tol=0.0001, verbose=0 | 0.6553 | 0.6148 | 0.6553| 0.6044 |
| #4 | (1,3) | C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=115, tol=0.0001, verbose=0 | 0.7284 | 0.7138 | 0.7284 | 0.7151 |

#### Multinomial Naive Bayes
| Multinomial Naive Bayes Model   | TFIDF n-gram Range | Model Parameters  | Accuracy  | Precision  | Recall | F-score|
| :-----: | :----: | :----------:      | :-------: | :--------: | :----: | :----: |
| #0 | (1,1) | alpha=1.0, class_prior=None, fit_prior=True | 0.5304 |  0.5471 | 0.5304 | 0.3790 | 
| #1 | (1,2) | alpha=0, class_prior=None, fit_prior=True | 0.6290 | 0.6287 | 0.6290 | 0.5984 |
| #2 | (1,1) | alpha=0, class_prior=None, fit_prior=True | 0.6111 | 0.5814 | 0.6111 | 0.5631 | 
| #3 | (1,2) | alpha=0, class_prior=None, fit_prior=False | 0.6197 | 0.6121 | 0.6197 | 0.5967 |

#### LSTM

| LSTM Model  | Number of Epoch | Max Number of Words | Max Input Length | Embedding Dimension | Batch Size  | Accuracy |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| #1 | 9 | 50,000 | 250 | 100 | 256 | 0.693 |
| #2 | 9 | 50,000 | 350 | 100 | 256 | 0.691 |
| #3 | 9 | 50,000 | 428 | 100 | 256 | 0.698 |

#### Bidirectional LSTM

| Bidirectional LSTM Model  | Number of Epoch | Max Number of Words | Max Input Length | Embedding Dimension | Batch Size  | Accuracy |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| #1 | 9 | 50,000 | 256 | 100 | 512 | 0.6897 |
| #2 | 9 | 50,000 | 256 | 100 | 256 | 0.6948 |
| #3 | 10 | 50,000 | 350 | 100 | 512 | 0.6859 |
| #4 | 9 | 50,000 | 350 | 100 | 256 |  0.6970 |
| #5 | 11 | 50,000 | 512 | 100 | 256 | 0.6957 |


#### Bidirectional LSTM with Self-Attention

| Bidirectional LSTM with Self-Attention Model | Number of Epoch | Max Number of Words | Max Input Length | Embedding Dimension | Batch Size  | Accuracy |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| #1 | 7 | 50,000 | 256 | 100 | 128 | 0.6948 |
| #2 | 7 | 50,000 | 350 | 100 | 128 | 0.6955|
| #3 | 7 | 50,000 | 512 | 100 | 64 | 0.6986 |
| #4 | 7 | 100,000| 512 | 256 | 64 | 0.7091 |

#### BERT

| BERT Model  | Number of Epoch | Max Input Length | Batch Size  | Accuracy |
| :----: | :----: | :----: | :----: | :----: |
| #1 | 15 | 25 | 256 | 0.6552 |
| #2 | 9 | 50 | 256 | 0.6836 |
| #3 | 10 | 75 | 128 | 0.7077 |
| #4 | 8 | 150 | 64 | 0.7344 |
| #5 | 9 | 200 | 64 | 0.7410 |
| #6 | 7 | 250 | 16 | 0.7487 |
| #7 | 10 | 300 | 16 | 0.7512 |
| #8 | 8 | 350 | 8 | 0.7552 |
| #9 | 10 | 428 | 8 | 0.7557 |
| #10 | 7 | 512 | 8 | 0.7580 |  


### Citation
* Cam P. Covid-19 Tweets EDA, Classification, BERT. Kaggle. Available at: https://www.kaggle.com/campudney/covid-19-tweets-eda-classification-bert/comments
* Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
* Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.
* Kevin, S. (2018). Effect of batch size on training dynamics. Medium. Available at: https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e
* Kowsari, K., Jafari Meimandi, K., Heidarysafa, M., Mendu, S., Barnes, L., & Brown, D. (2019). Text classification algorithms: A survey. Information, 10(4), 150.
* Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).
* Minaee, S., Kalchbrenner, N., Cambria, E., Nikzad, N., Chenaghlu, M., & Gao, J. (2020). Deep learning based text classification: A comprehensive review. arXiv preprint arXiv:2004.03705.
* Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. arXiv preprint arXiv:1802.05365
* Pietro, M., (2020). Text Classification With NLP: Tf-Idf Vs Word2vec Vs BERT. Medium. Available at: https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
* Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019, October). How to fine-tune bert for text classification?. In China National Conference on Chinese Computational Linguistics (pp. 194-206). Springer, Cham.
* Wang, S. I., & Manning, C. D. (2012, July). Baselines and bigrams: Simple, good sentiment and topic classification. In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 90-94).
* Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. In Advances in neural information processing systems (pp. 649-657).
