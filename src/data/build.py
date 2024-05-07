from pathlib import Path
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import random
import opensoundscape as opso
from opensoundscape.preprocess.utils import show_tensor_grid

BASE_PATH = Path.cwd().parents[1]
sys.path.append(str(BASE_PATH / "src"))  # for utils
sys.path.append(str(BASE_PATH / "src" / "data"))  # for download_recordings
from utils import *
import download_recordings


# Choose column contents when the dataframe is grouped by recording id
# instead of clip id.
# column name : aggregation method
recordings_metadata_dict = {
    "recording_url": "first",
    "task_method": "first",
    "project": "first",
    "detection_time": lambda x: list(x),
    "tag_duration": lambda x: list(x),
    "latitude": "first",
    "longitude": "first",
    "file_type": "first",
    "media_url": "first",  # for debugging
    "individual_order": "max",
    "location_id": "first",
}


# Show clips per task method
def report_counts(df: pd.DataFrame, header: str = ""):
    total_target_tags = len(
        df.loc[df.task_method.isin(["1SPT", "1SPM", "no_restrictions"])]
    )
    task_methods = df.task_method.value_counts(dropna=False)
    present_absent_counts = df.groupby("task_method", dropna=False).agg(
        {"target_present": "sum", "target_absent": "sum"}
    )
    present_clips = len(df.loc[df.target_present == True])
    absent_clips = len(df.loc[df.target_absent == True])

    print("\n--------------------------------------------------")
    print(header)
    print(f"clips per task method = \n {task_methods}")
    print(f"total clips = {len(df)}")
    print("\nclips generated from each tagging method:")
    print(present_absent_counts)
    print(f"total present clips =  {present_clips}")
    print(f"total absent clips =  {absent_clips}")
    print(f"total available human labelled tags = {total_target_tags}")
    return


# Split the dataset into training and validation sets
# This is done by location id, to ensure that the model generalizes to new areas.
# TODO - also split by date in case some ARUs are within earshot of each other.
def make_train_valid_split(df, seed=None, pct_train=0.8):
    """
    Makes a location based train/valid or train/test split of a dataframe.
    pct_train is the proportion of locations to include in the training set.
    """

    np.random.seed(seed)
    # Get unique location_ids and shuffle them
    unique_locations = df["location_id"].unique()
    np.random.shuffle(unique_locations)

    # Split the unique location_ids by pct_train e.g. 80% train, 20% valid
    split_index = int(len(unique_locations) * pct_train)
    train_locations = unique_locations[:split_index]
    valid_locations = unique_locations[split_index:]

    # Mark the rows in original DataFrame
    df["is_valid"] = df["location_id"].isin(valid_locations)

    # Split the DataFrame into two based on 'is_valid' then drop the column
    train_df = df[df["is_valid"] == False]
    valid_df = df[df["is_valid"] == True]
    train_df = train_df.drop(columns=["is_valid"])
    valid_df = valid_df.drop(columns=["is_valid"])

    return train_df, valid_df


def new_labelled_df(
    df: pd.DataFrame,
    target_species="OSFL",
    download_n: int = 0,
    sample_duration: float = 3.0,
    overlap_fraction: float = 0.5,
    existing_test_set: pd.DataFrame | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates absent and present tags for one species from a cleaned human labelled dataframe.

    The audio starts out as long recordings, which are downloaded from the recording_url field (not yet publicly available) in the input dataframe.

    Segements containing target or non-target audio are identified by the tag window onset and duration times along with the tagging method used for that recording. The audio is then split into clips, and the clips are labelled as present or absent.

    Finally the dataset is split into training and validation sets.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned human labelled dataframe
    target_species : str
        Species code for the target species
    download_n : int
        Number of recordings containing at least one target vocalization to download. Can be 0
    sample_duration : float
        window length in seconds
    overlap_fraction : float
        fraction of overlap between windows
    existing_test_set : pd.DataFrame
        Dataframe containing the test set. If provided, the locations in the test set will be removed from the training set.
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame] : training and validation dataframes


    """

    recording_path = BASE_PATH / "data" / "raw" / "recordings" / target_species

    # filter for target species
    target_df = df.loc[df.species_code == target_species]
    target_locations = target_df.location_id.unique()

    # keep columns specified in utils.py
    if drop_extra_cols:
        df = df[keep_cols]
        target_df = target_df[keep_cols]

    # group the dataframe by recording_id and aggregate the columns
    recordings = target_df.groupby("recording_id").agg(recordings_metadata_dict)

    # add filename column to the dataframe:
    # recording-<recording_id>.<file_type>
    recordings["filename"] = recordings.apply(
        lambda row: "recording-" + str(row.name) + "." + row.file_type, axis=1
    )

    # add column for relative path and set as index.
    # This enables AudioFileDataset to find the recordings.
    recordings["relative_path"] = recordings.apply(
        lambda row: Path("..")
        / ".."
        / "data"
        / "raw"
        / "recordings"
        / target_species
        / row.filename,
        axis=1,
    )
    recordings = recordings.set_index("relative_path")

    # get a list of the files that have already been downloaded
    # since we may have some downloaded already and we may need to download more,
    # make dataframes of downloaded and not downloaded reocrdings containing the target species.
    downloaded_recordings = [file.name for file in (recording_path.glob("*"))]
    df_downloaded_recordings = recordings.loc[
        recordings.filename.isin(downloaded_recordings)
    ]
    df_not_downloaded = recordings.loc[~recordings.filename.isin(downloaded_recordings)]
    print(f"{len(df_not_downloaded)} not downloaded")

    def download(n: int = 1):
        """
        Download n recordings from the list of recordings that have not been downloaded yet. Save them to data/raw/recordings/target_species
        """
        audio_save_path = audio_save_path = Path(
            BASE_PATH / "data" / "raw" / "recordings" / target_species
        )
        audio_save_path.mkdir(parents=True, exist_ok=True)
        download_recordings.from_url(
            df_not_downloaded, "recording_url", audio_save_path, target=None, n=n
        )

    download(download_n)

    ### Use opensoundscape methods to create split intervals

    # create a spectrogram preprocessor
    # force outputs to be same size
    pre = opso.SpectrogramPreprocessor(
        sample_duration=sample_duration, width=224, height=224
    )

    # re-index the dataframe with 'relative_path' as the index ready for AudioFileDataset and AudioSplittingDataset
    downloaded_paths_df = pd.DataFrame(df_downloaded_recordings.index).set_index(
        "relative_path"
    )

    splitting_dataset = opso.AudioSplittingDataset(
        downloaded_paths_df,
        pre,
        overlap_fraction=overlap_fraction,
        final_clip="full",
    )

    # remove mulit-index for adding labels
    clip_splits = splitting_dataset.label_df.reset_index()

    # merge the single indexed dataframe with the original data, so that each clip has the detection times necessary to calculate the label
    df = clip_splits.merge(df_downloaded_recordings, left_on="file", right_index=True)

    # filter for recordings tagged using 1SPT or 1SPM methods
    keep_recs = df.loc[
        (
            (df.task_method == "no_restrictions")  # no restrictions on tagging
            | (df.task_method == "1SPT")  # 1 sample per task
            | (df.task_method == "1SPM")  # 1 sample per minute
        )
    ].file
    df = df.loc[df.file.isin(keep_recs)]

    ### Create labels for each clip ###

    # calculate target present clips
    def clip_contains_target_tag(row: pd.Series):
        """
        Adds a row to dataframe indicating presence of target.
        If the detection onset plus duration falls within the clip, return 1. Else return 0.
        """
        start_time = row.start_time
        end_time = row.end_time
        detection_times = row.detection_time
        tag_durations = row.tag_duration
        for det, dur in zip(detection_times, tag_durations):
            if det >= start_time and det + dur <= end_time:
                return float(1)
        return float(0)

    df["target_present"] = df.apply(clip_contains_target_tag, axis=1)

    # calculate target absence
    def clip_is_before_first_tag(row):
        """
        Returns 1 if the end of the clip is before the start of the first detection.
        """
        end_time = row.end_time
        detection_time = row.detection_time

        if end_time < detection_time[0]:
            return float(1)
        else:
            return float(0)

    df["target_absent"] = df.apply(clip_is_before_first_tag, axis=1)

    # filter out the rest of the clips because these are made up from
    # *partial overlap*: audio containing only part of the target call. If we use an overlap of 0.5 at inference, and the window is longer than the max target length, then we should always have at least one clip that contains the whole target call during inference.
    # - *audio from after the first target tag*: this might contain unlabeled target calls.
    df = df.drop(
        df.loc[df.target_present == False].loc[df.target_absent == False].index
    )

    # Set multi index for use with opensoundscape
    df.set_index(["file", "start_time", "end_time"], inplace=True)

    train_df, valid_df = make_train_valid_split(df, seed)

    # Drop the locations found in the premade test set from the training set
    if existing_test_set is not None:
        test_locations = existing_test_set.location_id.unique()
        train_locations = train_df.location_id.unique()
        # find intersect
        intersect = list(set(test_locations) & set(train_locations))
        train_df = train_df[~train_df.location_id.isin(intersect)]
        print(f"dropped {len(intersect)} locations from training set")

    report_counts(train_df, "train set")
    report_counts(valid_df, "valid set")

    return train_df, valid_df


# Separately download samples from outside of the target habitat. These  will all be negative examples.
def other_habitat_df(df, target_species="OSFL", download_n=0, seed=None):
    """
    Download recordings from outside of the target habitat to use as negative examples.
    This is simply a random sample across all locations where the target species was not detected.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned human labelled dataframe
    target_species : str
        Species code for the target species to filter out of this set of recordings
    out_of_habitat_n : int
        Number of recordings from outside of the target habitat to download. Can be 0
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame : dataframe containing recordings from outside of the target habitat
    """
    recording_path = (
        BASE_PATH / "data" / "raw" / "recordings" / f"{target_species}_other_habitats"
    )

    # filter for target species
    target_df = df.loc[df.species_code == target_species]
    target_locations = target_df.location_id.unique()

    # find habitats where target was never detected
    other_habitat_df = df.loc[~df.location_id.isin(target_locations)]

    # keep columns specified in utils.py
    df = df[keep_cols]
    other_habitat_df = other_habitat_df[keep_cols]

    # group the dataframe by recording_id and aggregate the columns
    recordings = other_habitat_df.groupby("recording_id").agg(recordings_metadata_dict)

    # add filename column to the dataframe:
    # recording-<recording_id>.<file_type>
    recordings["filename"] = recordings.apply(
        lambda row: "recording-" + str(row.name) + "." + row.file_type, axis=1
    )

    # add column for relative path and set as index.
    # This enables AudioFileDataset to find the recordings.
    recordings["relative_path"] = recordings.apply(
        lambda row: Path("..")
        / ".."
        / "data"
        / "raw"
        / "recordings"
        / f"{target_species}_other_habitats"
        / row.filename,
        axis=1,
    )
    recordings = recordings.set_index("relative_path")

    # get a list of the files that have already been downloaded
    # since we may have some downloaded already, and we may need to download more, make dataframes of downloaded and not downloaded reocrdings containing the target species.
    downloaded_recordings = [file.name for file in (recording_path.glob("*"))]
    df_downloaded_recordings = recordings.loc[
        recordings.filename.isin(downloaded_recordings)
    ]
    df_not_downloaded = recordings.loc[~recordings.filename.isin(downloaded_recordings)]
    print(f"{len(df_not_downloaded)} not downloaded")

    def download(n: int = 1):
        """
        Download n recordings from the list of recordings that have not been downloaded yet. Save them to data/raw/recordings/target_species
        """
        audio_save_path = audio_save_path = Path(
            BASE_PATH
            / "data"
            / "raw"
            / "recordings"
            / f"{target_species}_other_habitats"
        )
        audio_save_path.mkdir(parents=True, exist_ok=True)
        download_recordings.from_url(
            df_not_downloaded, "recording_url", audio_save_path, target=None, n=n
        )

    download(download_n)

    return df_downloaded_recordings
