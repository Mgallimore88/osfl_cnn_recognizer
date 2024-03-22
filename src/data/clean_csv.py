# Preprocessing script to load raw csv file and create a modified, cleaned DataFrame as output.


def process_raw_csv(path_to_raw_csv, clean_all: bool = True):
    """
    Gets from raw training data csv file to cleaned metadata pkl file.
    set clean_all to False to skip over some assumptions made about the data - namely that the TMTT and call vocalizations should be removed.

    - Load raw csv file
    - Drop last entry since it's all NaN values.
    - Replace empty fields with -1 for verifier_id to enable import to pandas dataframe as int type.
    - Change all the data types in the DataFrame to the types specified in in the preset_types.py
    - Drop 'too many to tag' abundance tags.
    - Drop non song vocalizations
    - Drop recordings not labeled in wildtrax
    - Remove the clips which don't contain a link to a clip
    - Remove any clips which belong to a recording with a missing recording_url
    - Remove clips from projects which might contain data which contaminates the dataset with duplicated or synthetic recordings.
    - Remove duplicated clips from the database
    - Add a column to store file type derived from clip URL
    - Export the cleaned version of the database
    """
    import pandas as pd
    from pathlib import Path
    from preset_types import type_dict

    print("Processing raw csv file...")
    data_path = Path("../../data/")

    # Load the raw csv file
    if path_to_raw_csv is None:
        df = pd.read_csv(
            data_path / "raw" / "TrainingData_BU&Public_CWS_with_rec_links.csv",
            low_memory=False,
        )
    else:
        df = pd.read_csv(path_to_raw_csv, low_memory=False)

    if clean_all:
        # Drop 'too many to tag' abundance tags.
        tmtt_idxs = df[df.abundance == "TMTT"].index
        df.drop(tmtt_idxs, inplace=True)

        # Drop non song vocalizations
        not_song_idxs = df[df.vocalization != "Song"].index
        df.drop(not_song_idxs, inplace=True)

    # Drop last entry since it was all NaN values when inspected.
    df.drop(df.tail(1).index, inplace=True)

    # Replace empty fields with -1 for
    df.loc[df["verifier_id"].isna(), "verifier_id"] = -1

    # Specify the task method "No Restrictions" instead of inferring if from the n/a value type
    df.loc[df["task_method"].isna(), "task_method"] = "no_restrictions"

    # Change all the data types in the DataFrame to the types specified in in the preset_types.py
    df = df.astype(type_dict)

    # Drop recordings not labeled in wildtrax
    labeled_elsewhere_idxs = df[df.tagged_in_wildtrax == "f"].index
    df.drop(labeled_elsewhere_idxs, inplace=True)

    # Remove the clips which don't contain a link to a clip
    df.drop(df.loc[df.clip_url == "nan"].index, inplace=True)

    # Remove clips from projects which might contain duplicated or synthetic recording data.
    removed_projects = [
        "2023 Playback Experiment",
        "ARU Test Project Model Comparisons 2021",
        "CWS-Ontario Birds of James Bay Lowlands 2021 (Resample)",
        "Wildtrax Demo 2020",
    ]
    df.drop(df.loc[df.project.isin(removed_projects)].index, inplace=True)

    # Remove any clips which belong to a recording with a missing recording_url
    df.drop(df.loc[df.recording_url == "nan"].index, inplace=True)

    def filter_duplicate_clips(df: pd.DataFrame) -> pd.DataFrame:
        """Filter out duplicate clips based on tag_id"""
        return df.loc[df.tag_id.drop_duplicates().index]

    df = filter_duplicate_clips(df)

    # Add a column to store file type derived from clip URL
    # df["file_type"] = None
    def get_file_type(url: str) -> str:
        return url.split(".")[-1]

    df["file_type"] = df.recording_url.apply(get_file_type)

    # Export the cleaned version of the database
    interim_data_path = Path("../../data/interim/")
    df.to_pickle(interim_data_path / "cleaned_metadata.pkl")

    print(
        "Done processing raw csv file. Outputted to data/interim/cleaned_metadata.pkl"
    )


if __name__ == "__main__":
    process_raw_csv(input("Enter path to raw csv file: "), clean_all=True)
