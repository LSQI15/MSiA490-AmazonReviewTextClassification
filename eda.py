import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.stdout = open('EDA/EDA_text.txt', 'w')


def read_data(file_path):
    """
    function to read in pre-processed data for doing eda
    :param file_path: path to the input data file
    :return: a pandas dataframe object
    """
    df = pd.read_csv(file_path)
    return df


def dataset_stats(df):
    """
    function to generate summary statistics for the dataset
    :param df: data frame containing pre-processed data
    :return: None
    """
    # preview the first 10 rows
    print(df.head(10))
    # Number of reviews
    print('\nNumber of reviews: ' + str(df.shape[0]))
    # Average length of reviews
    print('\nAverage length of reviews: ' + str(df['reviewText'].apply(len).mean()))
    # Number of labels
    print('\nNumber of unique labels: ' + str(len(df['score'].unique())))


def overall_score_eda(df, figure_path):
    """
    function to do eda for the distribution of overall score
    :param df: a pandas dataframe object containing pre-processed data
    :param figure_path: path to the output figure
    :return: None
    """
    # values counts for each review score
    print('\nValue counts for each review score (1-5)')
    print(df['score'].value_counts())
    # visualize review scores distribution
    fig, ax = plt.subplots()
    df['score'].value_counts().plot(ax=ax, kind='bar')
    total = float(len(df['score']))
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() - 0.45
        y = p.get_height()
        ax.annotate(percentage, (x, y))
    plt.title('Distribution of Review Scores')
    plt.xlabel('Review Score')
    plt.ylabel('Number of Reviews')
    plt.savefig(figure_path)


def main():
    """
    main function to do exploratory data analysis
    :return: None
    """
    in_file_path = 'Data/kindle_store_reviews.csv'
    score_distribution_graph = 'EDA/score_distribution_graph.png'

    df = read_data(in_file_path)
    df['reviewText'] = df['reviewText'].astype(str)
    dataset_stats(df)
    overall_score_eda(df, score_distribution_graph)


if __name__ == "__main__":
    main()
