# Preprocessing script to load raw csv file and create a modified, cleaned DataFrame as output.


def process_raw_csv():
    """
    Gets from raw training data csv file to cleaned metadata csv file.

    1. Load raw csv file
    2. Drop last entry since it's all NaN values.
    3. Replace empty fields with -1 for verifier_id to enable import to pandas dataframe as int type.
    4. Change all the data types in the DataFrame to the types specified in in the preset_types.py
    5. Drop 'too many to tag' abundance tags.
    6. Drop non song vocalizations
    7. Drop recordings not labeled in wildtrax
    8. Remove the clips which don't contain a link to a clip
    9. Add a column to store file type derived from clip URL
    10. Export the cleaned version of the database
    """
    import pandas as pd
    import re
    from pathlib import Path
    from preset_types import type_dict

    print("Processing raw csv file...")
    data_path = Path("../../data/")

    # Load the raw csv file
    meta = pd.read_csv(data_path / "raw/TrainingData_BU&Public_CWS_with_rec_links.csv")

    # Drop last entry since it's all NaN values.
    meta.drop(meta.tail(1).index, inplace=True)

    # Replace empty fields with -1 for
    meta.loc[meta["verifier_id"].isna(), "verifier_id"] = -1

    # Change all the data types in the DataFrame to the types specified in in the preset_types.py
    meta = meta.astype(type_dict)

    # Drop 'too many to tag' abundance tags.
    tmtt_idxs = meta[meta.abundance == "TMTT"].index
    meta.drop(tmtt_idxs, inplace=True)

    # Drop non song vocalizations
    not_song_idxs = meta[meta.vocalization != "Song"].index
    meta.drop(not_song_idxs, inplace=True)

    # Drop recordings not labeled in wildtrax
    labeled_elsewhere_idxs = meta[meta.tagged_in_wildtrax == "f"].index
    meta.drop(labeled_elsewhere_idxs, inplace=True)

    # Remove the clips which don't contain a link to a clip
    meta.drop(meta.loc[meta.clip_url == "nan"].index, inplace=True)

    # Add a column to store file type derived from clip URL
    # meta["file_type"] = None
    def get_file_type(url: str) -> str:
        return url.split(".")[-1]

    has_url = ~(meta.clip_url == "nan")
    meta["file_type"] = meta.recording_url.apply(get_file_type)

    # Export the cleaned version of the database
    interim_data_path = Path("../../data/interim/")
    meta.to_csv(interim_data_path / "processed_metadata.csv")
    meta.to_pickle(interim_data_path / "processed_metadata.pkl")

    print(
        "Done processing raw csv file. Outputted to data/interim/processed_metadata.csv + pkl"
    )


if __name__ == "__main__":
    process_raw_csv()
