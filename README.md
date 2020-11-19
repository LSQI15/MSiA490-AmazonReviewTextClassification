### Northwestern University
### MSiA-490 Fall 2020
### Amazon Kindle Store Review Text Classification Project
### Siqi Li

<!-- toc -->

## Repo structure

<!-- toc -->

- [Project Topic](#project-topic)
- [Dataset](#dataset)
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
For this project, I conducted the comparison of 7 multi-class text classification approaches (logistic regression, 
support vector machine (SVM), multinomial naive bayes, BERT, LSTM, bidirectional LSTM, and bidirectional LSTM with 
self-attention). Specifically, I will train each of the model on the training set (80% of data) and evaluate their 
performances on the test set (20% of data) using metrics such as accuracy, precision, recall, and F-1 score. The best 
model will then be productized to take review text(s) as input and the output corresponding predicted review score (1-5).

### Dataset
The dataset chosen for this project is Amazon video review data from [Amazon Review Data](https://snap.stanford.edu/data/web-Amazon.html). 
The raw dataset contains information such as reviews (ratings, text, helpfulness votes) and product metadata 
(descriptions, category information, price, brand, and image features) for 717,651 reviews. Due to computational 
limitation, only the first 500,000 records are used in this text classification project. Each record has the overall 
review score (integer from 1 to 5) and the review text (string)
    
### Model Results

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
