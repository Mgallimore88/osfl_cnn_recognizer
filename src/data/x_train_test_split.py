"""
Python script to load the processed dataframe and split it into train and test sets. 

The split is done randomly by location id, so that performance metrics always reflect the model's performance on unseen locations. This is also a safeguard against data leakage, in case recordings from one location were uploaded twice by mistake.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data():
    BASE_PATH = Path.cwd().parents[1]
    data_path = BASE_PATH / "data"
    df = pd.read_csv(data_path / "interim" / "processed_metadata.csv")

    Path.mkdir(data_path / "processed" / "test_set", parents=True, exist_ok=True)
    Path.mkdir(data_path / "processed" / "train_set", parents=True, exist_ok=True)

    # Take a random sample from 20% of the locations without replacement
    locations = df.location_id.unique()
    train_locations, test_locations = train_test_split(
        locations, test_size=0.2, random_state=42
    )

    df_test = df.loc[df.location_id.isin(test_locations)]
    df_train = df.loc[df.location_id.isin(train_locations)]

    # Save the test df and the training df to file
    print("Saving test and train sets to file...")
    df_test.to_pickle(data_path / "processed" / "test_set" / "test_set.pkl")
    df_train.to_pickle(data_path / "processed" / "train_set" / "train_set.pkl")


if __name__ == "__main__":
    split_data()
