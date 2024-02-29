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


# column name : aggregation method
# For choosing column contents wheh the dataframe is grouped by recording id
# instead of clip id.
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


def dataset_from_df(
    df: pd.DataFrame,
    target_species="OSFL",
    download_n: int = 0,
    one_class: bool = False,
) -> tuple[opso.AudioFileDataset, opso.AudioFileDataset, pd.Series, pd.Series]:
    """
    Returns a labelled dataset from a cleaned dataframe.
    The audio starts out as long recordings, which are downloaded from the recording_url field in the dataframe (not yet publicly available). Segements containing target or non-target audio are identified by the tag window onset and duration times along with the tagging method used for that recording. The audio is then split into clips, and the clips are labelled as containing target or non-target audio. The clips are then passed into an AudioFileDataset, which returns a spectrogram tensor and a label tensor for each clip.
    Finally the dataset is split into training and validation sets.
    """

    # load data including species timestamps for label calculation
    df_full = pd.read_pickle(
        BASE_PATH
        / "data"
        / "interim"
        / "train_and_valid_set"
        / "train_and_valid_set.pkl"
    )

    recording_path = BASE_PATH / "data" / "raw" / "recordings" / target_species

    # keep columns specified in utils.py
    target_df = df.loc[df.species_code == target_species]
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
    pre = opso.SpectrogramPreprocessor(sample_duration=3.0, width=224 * 3, height=224)

    # re-index the dataframe with 'relative_path' as the index ready for AudioFileDataset and AudioSplittingDataset
    downloaded_paths_df = pd.DataFrame(df_downloaded_recordings.index).set_index(
        "relative_path"
    )

    splitting_dataset = opso.AudioSplittingDataset(
        downloaded_paths_df,
        pre,
        overlap_fraction=0.5,
        final_clip="full",
    )

    # remove mulit-index for adding labels
    clip_splits = splitting_dataset.label_df.reset_index()

    # merge the single indexed dataframe with the original data, so that each clip has the detection times necessary to calculate the label
    df = clip_splits.merge(df_downloaded_recordings, left_on="file", right_index=True)

    # filter for recordings tagged using 1SPT or 1SPM methods
    keep_recs = df.loc[
        (
            df.task_method.isna()  # no restrictions on tagging
            | (df.task_method == "1SPT")
            | (df.task_method == "1SPM")
        )
    ].file
    df = df.loc[df.file.isin(keep_recs)]

    ### Create labels for each clip ###

    # calculate target presence
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

    df["target_presence"] = df.apply(clip_contains_target_tag, axis=1)

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

    df["target_absence"] = df.apply(clip_is_before_first_tag, axis=1)

    def report_counts(df: pd.DataFrame, info: str = ""):
        total_target_tags = len(
            target_df.loc[target_df.task_method.isin(["1SPT", "1SPM", np.nan])]
        )
        task_methods = df.task_method.value_counts(dropna=False)
        presence_absence_counts = df.groupby("task_method", dropna=False).agg(
            {"target_presence": "sum", "target_absence": "sum"}
        )
        targets = len(df.loc[df.target_presence == True])
        absences = len(df.loc[df.target_absence == True])
        undefined = len(
            df.loc[(df.target_presence == False)].loc[(df.target_absence == False)]
        )

        print("\n--------------------------------------------------")
        print(info)
        print(f"recordings per task method = \n {task_methods}")
        print(f"total recordings = {len(df)}")
        print("\nTags generated from each tagging method:")
        print(presence_absence_counts)
        print(f"total target clips =  {targets}")
        print(f"total absence clips =  {absences}")
        print(f"total available human labelled target tags = {total_target_tags}")
        print(f"undefined {undefined}")

        return

    report_counts(df, "before filtering undefined clips")

    # filter out the rest of the clips because these are made up from
    # - partial overlap: audio containing only part of the target call. If we use an overlap of 0.5 at inference, and the window is longer than the max target length, then we should always have at least one clip that contains the whole target call during inference.
    # - audio from after the first target tag: this might contain unlabeled target calls.
    df = df.drop(
        df.loc[df.target_presence == False].loc[df.target_absence == False].index
    )

    # Set multi index for passing into AudioFileDataset
    df.set_index(["file", "start_time", "end_time"], inplace=True)

    report_counts(df, "after filtering undefined clips")

    # Split the dataset into training and validation sets
    def make_train_valid_split(df):
        # Get unique location_ids and shuffle them
        unique_locations = df["location_id"].unique()
        np.random.shuffle(unique_locations)

        # Split the unique location_ids (80% train, 20% valid)
        split_index = int(len(unique_locations) * 0.80)
        train_locations = unique_locations[:split_index]
        valid_locations = unique_locations[split_index:]

        # Mark the rows in original DataFrame
        df["is_valid"] = df["location_id"].isin(valid_locations)

        # Split the DataFrame into two based on 'is_valid'
        train_df = df[df["is_valid"] == False]
        valid_df = df[df["is_valid"] == True]

        return train_df, valid_df, train_locations, valid_locations

    train_df, valid_df, train_locations, valid_locations = make_train_valid_split(df)

    if one_class:
        train_ds = opso.AudioFileDataset(
            train_df[["target_presence"]],
            pre,
        )
        valid_ds = opso.AudioFileDataset(
            valid_df[["target_presence"]],
            pre,
            bypass_augmentations=True,  # remove preprocessing for validation set
        )
    else:
        train_ds = opso.AudioFileDataset(
            train_df[["target_absence", "target_presence"]],
            pre,
        )
        valid_ds = opso.AudioFileDataset(
            valid_df[["target_absence", "target_presence"]],
            pre,
            bypass_augmentations=True,  # remove preprocessing for validation set
        )

    display_sample_idxs = random.sample(range(len(train_df)), 5)

    sample_tensors = [train_ds[i].data for i in display_sample_idxs]
    sample_labels = [train_ds[i].labels.target_presence for i in display_sample_idxs]

    _unused_fig = show_tensor_grid(sample_tensors, 2, labels=sample_labels)

    return train_ds, valid_ds, train_locations, valid_locations


if __name__ == "__main__":
    dataset_from_df(df, target_species="OSFL", download_n=0)
