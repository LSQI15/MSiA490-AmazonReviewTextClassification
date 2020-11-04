import re
import pandas as pd


def clean_review(review):
    """
    function to clean raw text reviews by removing special characters, extra space, punctuations, etc
    :param review: a raw review of type string
    :return: a cleaned review
    """
    review = review.lower()
    replace_by_space = re.compile('[/(){}\[\]\|@,;]')
    remove_special_characters = re.compile('[^0-9a-z +]')
    remove_extra_space = re.compile('[^A-Za-z0-9]+')
    remove_numbers = re.compile('[0-9]+')
    review = re.sub(replace_by_space, ' ', review)
    review = re.sub(remove_special_characters, ' ', review)
    review = re.sub(remove_numbers, ' ', review)
    review = re.sub(remove_extra_space, ' ', review)
    return review.strip()


def process(df, column):
    """
    function to process all raw reviews in the data frame
    :param df: a preprocessed pandas data frame
    :param column: the name of the column that stores reviews
    :return: a data frame contains processed reviews
    """
    df[column] = df[column].apply(clean_review)
    return df


def main():
    """
    main function to process the preprocessed data
    :return: None
    """
    in_file_path = 'Data/video_reviews.csv'
    out_file_path = 'Data/processed_video_reviews.csv'
    df = pd.read_csv(in_file_path).dropna()  # some reviews have only score, so drop them
    processed_df = process(df, 'reviewText')  # clean all reviews
    processed_df.to_csv(out_file_path, index=False)


if __name__ == "__main__":
    main()