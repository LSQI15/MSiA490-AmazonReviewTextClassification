import pandas as pd
import gzip


def parse(filename):
    """
    function to parse the raw data
    :param filename: path to raw data file
    :return: a dictionary containing all raw data
    """
    f = gzip.open(filename, 'r')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = str(l).find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        eName = l[:colonPos - 2].decode('utf-8')
        rest = l[colonPos:].decode('utf-8')
        entry[eName] = rest
    yield entry


def data_preprocess(filename, columns_to_keep):
    """
    main function to process the raw data and return raw data in a pandas df
    :param filename: path to raw data file
    :param columns_to_keep: a list of columns to keep
    :return: a pandas data frame containing raw data
    """
    # add all review to a list
    review = []
    for e in parse(filename):
        review.append(e)
    # convert records to a pandas data frame
    review_df = pd.DataFrame.from_records(review)
    review_df = review_df.rename(columns={'review/score': "score", 'review/text': 'reviewText'})
    review_df = review_df[columns_to_keep].dropna()
    # convert review score to integer type
    review_df['score'] = pd.to_numeric(review_df['score']).astype(int)
    return review_df


def main():
    """
    main function to pre-process the raw data (convert from json to pandas data frame,
    select columns to keep and drop missing values)
    :return: None
    """
    in_file_path = '/Users/siqili/Downloads/Amazon_Instant_Video.txt.gz'
    out_file_path = '../Data/video_reviews.csv'
    columns_to_keep = ['score', 'reviewText']
    preprocessed_data = data_preprocess(in_file_path, columns_to_keep)
    preprocessed_data.to_csv(out_file_path, index=False)


if __name__ == "__main__":
    main()
