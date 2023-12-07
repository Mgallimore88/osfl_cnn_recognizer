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
}


def dataset_from_df(
    df: pd.DataFrame, target_species="OSFL", download_n: int = 0
) -> opso.AudioFileDataset:
    """
    Returns a labelled dataset from a cleaned dataframe.
    The audio starts out as long recordings, which are downloaded from the recording_url field in the dataframe (not yet publicly available). Segements containing target or non-target audio are identified by the tag window onset and duration times along with the tagging method used for that recording. The audio is then split into clips, and the clips are labelled as containing target or non-target audio. The clips are then passed into an AudioFileDataset, which returns a spectrogram tensor and a label tensor for each clip.
    """

    # load data including species timestamps for label calculation
    df_full = pd.read_pickle(
        BASE_PATH / "data" / "processed" / "train_set" / "train_set.pkl"
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
    pre = opso.SpectrogramPreprocessor(sample_duration=3.0)

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
    spt_recs = df.loc[(df.task_method == "1SPT") | (df.task_method == "1SPM")].file
    df = df.loc[df.file.isin(spt_recs)]

    ### Create labels for each clip ###

    # calculate target presence
    def clip_contains_target_tag(row: pd.Series):
        """
        Adds a row to dataframe indicating presence of target.
        If the detection onset plus duration falls within the clip, return True. Else return False.
        """
        start_time = row.start_time
        end_time = row.end_time
        detection_time = row.detection_time
        tag_duration = row.tag_duration
        for det, dur in zip(detection_time, tag_duration):
            if det >= start_time and det + dur <= end_time:
                return True
        return False

    df["target_presence"] = df.apply(clip_contains_target_tag, axis=1)

    # calculate target absence
    def clip_is_before_first_tag(row):
        """
        Returns True if the end of the clip is before the start of the first detection.
        """
        end_time = row.end_time
        detection_time = row.detection_time

        if end_time < detection_time[0]:
            return True
        else:
            return False

    df["target_absence"] = df.apply(clip_is_before_first_tag, axis=1)

    # filter out the rest of the clips because these are made up from
    # - partial overlap: audio containing only part of the target call. If we use an overlap of 0.5 at inference, and the window is longer than the max target length, then we should always have one clip that contains the whole target call during inference.
    # - audio from after the first target tag: this might contain unlabeled target calls.
    df = df.drop(df.loc[df.target_presence == False][df.target_absence == False].index)

    # Set multi index for passing into AudioFileDataset
    df.set_index(["file", "start_time", "end_time"], inplace=True)

    audio_ds = opso.AudioFileDataset(
        df[["target_presence", "target_absence"]],
        pre,
    )

    sample_idxs = random.sample(range(len(audio_ds)), 5)

    tensors = [audio_ds[i].data for i in sample_idxs]
    labels = [audio_ds[i].labels.target_presence for i in sample_idxs]

    _ = show_tensor_grid(tensors, 2, labels=labels)

    return audio_ds


if __name__ == "__main__":
    build_labelled_dataset(df, target_species="OSFL", download_n=0)
