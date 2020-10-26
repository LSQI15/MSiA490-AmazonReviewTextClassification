### Northwestern University
### MSiA-490 Fall 2020
### Amazon Kindle Store Review Text Classification Project
### Siqi Li

<!-- toc -->

#### Project Topic
For this project, I will conduct the comparison of three multi-class text classification approaches (BERT, LSTM, fastText).
Specifically, I will train each of the model on the training set (80% of data) and evaluate their performance on the test 
set (20% of data) using metrics such as accuracy, precision, recall, and F-1 score. The best model will then be productized
to take review text(s) as input and the output corresponding predicted review score (1-5).

#### Dataset
The dataset chosen for this project is Kindle Store Review data from [Amazon Review Data (2018)](http://deepyeti.ucsd.edu/jianmo/amazon/index.html). 

This dataset is an updated version of the Amazon review dataset released in 2014. The raw dataset contains information 
such as reviews (ratings, text, helpfulness votes) and product metadata (descriptions, category information, price, 
brand, and image features) for 5,722,988 reviews in Amazon Kindle Store.

Due to computational limitation, only the first 1,000,000 records are used in this text classification project. Each 
record has the overall review score (integer from 1 to 5), the review text (string), and the datetime (yyyy-mm-dd) on 
which the review was written.

Link to dataset: http://deepyeti.ucsd.edu/jianmo/amazon/index.html

#### Literature Survey

Text Classification is the process of assigning tags or categories to text based on its content. It's one of the 
fundamental tasks in Natural Language Processing (NLP) and has a wide range of applications such as sentiment analysis 
and spam detection.

There are two main types of machine learning model for doing text classification:

* Traditional machine learning models:
    * Models such as logistic regression, support vector machines, naive bayes. 
    * This type of models leverage feature representations such as Bag-of-Words (with Tf-Idf) and Word Embedding (with 
    Word2Vec)
* Deep Learning based machine learning models:
    * Models such as Recurrent Neural Networks (RNNs), Convolutional Neural Networks (CNNS), Long Short Term Memory 
    (LSTM), and language models such as Bidirectional Encoder Representations from Transformers (BERT)
    * Languages models overcome the biggest limitation of the classic Word Embedding approach: polysemy disambiguation, 
    a word with different meanings (e.g. “ bank” or “stick”) is identified by just one vector.
        * ELMO (2018), which doesn’t apply a fixed embedding, looks at the entire sentence and then assigns an embedding
         to each word using a bidirectional LSTM. 
        * Google’s BERT (Bidirectional Encoder Representations from Transformers, 2018) combines ELMO context embedding 
        and several Transformers. Moreover, BERT is bidirectional, which was a big novelty for Transformers. The vector 
        BERT assigns to a word is a function of the entire sentence. Hence, a word can have different vectors based on 
        its contexts 

The commonly used evaluation metrics are:

    * Accuracy: the percentage of texts that were predicted with the correct label.
    * Precision: the number of true positives over the number of true positives plus the number of false positives.
    * Recall: the number of true positives over the number of true positives plus the number of false negatives
    * F1 Score: the harmonic mean of precision and recall.
    
In the case of multi-class classification, to evaluate the overall model performance, we need to conduct either 
micro-average or macro-average:

    * A macro-average will compute the metric independently for each class and then take the average (gives each class equal weight).
    * A micro-average will aggregate the contributions of all classes to compute the average metric (gives each classification instance equal weight).

#### Citation

* Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019, October). How to fine-tune bert for text classification?. In *China National Conference on Chinese Computational Linguistics* (pp. 194-206). Springer, Cham.
* Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.
* Minaee, S., Kalchbrenner, N., Cambria, E., Nikzad, N., Chenaghlu, M., & Gao, J. (2020). Deep learning based text classification: A comprehensive review. *arXiv preprint arXiv:2004.03705*.
* Pietro, M., (2020). Text Classification With NLP: Tf-Idf Vs Word2vec Vs BERT. *Medium*. Available at: <https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794>
* Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. In *Advances in neural information processing systems* (pp. 649-657).
* Kim, Y. (2014). Convolutional neural networks for sentence classification. *arXiv preprint arXiv:1408.5882*.
* Wang, S. I., & Manning, C. D. (2012, July). Baselines and bigrams: Simple, good sentiment and topic classification. In *Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* (pp. 90-94).
* Ni, J., Li, J., & McAuley, J. (2019, November). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 188-197).

