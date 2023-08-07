import pandas as pd

def load_data_from_csv(path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a Pandas DataFrame.
    Csv file requirements:
        - `user_id` - int
        - `app_id` - int
        - `is_recommended` - int [0/1]

    Parameters:
    - path (str): The file path of the CSV file to load.

    Returns:
    - df (pd.DataFrame): The loaded data as a Pandas DataFrame.
    """
    df = pd.read_csv(path, index_col=[0])
    return df