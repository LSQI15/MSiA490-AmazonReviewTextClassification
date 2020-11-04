import pandas as pd


def preprocess_data(in_file_path, out_file_path, columns_to_keep):
    """
    function to pre-process the raw data;
    convert unix time to datetime; keep only the first nrows rows with selected columns_to_keep;
    :param in_file_path: the path to the raw data file
    :param out_file_path: the path to the output data file
    :param columns_to_keep: a list of columns to keep
    :param nrows: number of rows to keep
    :return: None
    """
    raw = pd.read_json(in_file_path, compression='infer', lines=True)
    raw['reviewTime'] = pd.to_datetime(raw['unixReviewTime'], unit='s')
    filtered_df = raw[raw['verified'] == True][columns_to_keep]
    filtered_df = filtered_df.dropna()
    filtered_df = filtered_df.rename(columns={"overall": "score"})
    filtered_df.to_csv(out_file_path, index=False)


def main():
    """
    main function to pre-process the raw data
    :return: None
    """
    in_file_path = '/Users/siqili/Downloads/Kindle_Store.json.gz'
    out_file_path = 'Data/kindle_store_reviews.csv'
    columns_to_keep = ['overall', 'reviewText']
    preprocess_data(in_file_path, out_file_path, columns_to_keep)


if __name__ == "__main__":
    main()
