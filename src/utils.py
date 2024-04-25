import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from IPython.display import display
import random
import opensoundscape as opso
from opensoundscape.preprocess.utils import show_tensor_grid
import torch
import hashlib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path


### Pandas ###
def display_all(df: pd.DataFrame, max_rows: int = 0, max_columns: int = 0) -> None:
    """
    Display all columns and rows of a dataframe without truncating.
    """
    if max_rows == 0:
        max_rows = len(df)
    if max_columns == 0:
        max_columns = len(df.columns)
    with pd.option_context(
        "display.max_rows", max_rows, "display.max_columns", max_columns
    ):
        display(df)


### project dataframe ###
keep_cols = [
    "organization",
    "project",
    "project_id",
    "location_id",
    "recording_id",
    "recording_date_time",
    "species_code",
    "species_common_name",
    "detection_time",
    "task_duration",
    "tag_duration",
    "tag_id",
    "clip_url",
    "recording_url",
    "task_method",
    "latitude",
    "longitude",
    "file_type",
    "media_url",
    "individual_order",
]


### Error checking
def calculate_file_durations(df):
    # Takes multi indexed df with file paths as first index. Returns file durations.
    audio_files = df.index.get_level_values("file").unique().values
    opso.Audio.from_file(audio_files[0]).duration
    durations = []
    for file in audio_files:
        audio = opso.Audio.from_file(file)
        durations.append(audio.duration)
    return durations


def clean_confidence_cats(df, clean_unchecked=False):
    # Re-label the mis-labelled clips
    df.loc[df["confidence_cat"] == 5, "target_presence"] = 1.0
    df.loc[df["confidence_cat"] == 6, "target_presence"] = 0.0

    # drop the clips with confidence 1 or 2 since these were hard to label and wouldn't constitute clear examples of the target class.
    df = df[df["confidence_cat"] != 1]
    df = df[df["confidence_cat"] != 2]
    if clean_unchecked:
        # Drop the unverified clips
        df = df[df["confidence_cat"] != 0]
    return df


def show_sample_from_df(df: pd.DataFrame, label: str = "present"):
    """
    Play audio and plot spectrogram for an item in a dataframe.
    Index must be a multi index of path, offset, end time.
    args: df: dataframe with multi index
    label: "present" or "absent"
    """
    if label == "present":
        sample = df.loc[df.target_present == 1].sample()
    elif label == "absent":
        sample = df.loc[df.target_present == 0].sample()

    path, offset, end_time = sample.index[0]
    duration = end_time - offset
    audio = opso.Audio.from_file(path, offset=offset, duration=duration)
    spec = opso.Spectrogram.from_audio(audio)
    audio.show_widget()
    spec.plot()


def show_index_from_df(df, idx):
    """
    Play audio and plot spectrogram for an item in a dataframe.
    Index must be a multi index of path, offset, end time.
    args: df: dataframe with multi index
    idx: index of the item to show
    """
    path, offset, end_time = df.index[idx]
    duration = end_time - offset
    audio = opso.Audio.from_file(path, offset=offset, duration=duration)
    spec = opso.Spectrogram.from_audio(audio)
    audio.show_widget()
    spec.plot()


def plot_metrics_across_thresholds(
    df, preds_column: str = "present_pred", title="Metrics across thresholds"
):
    """
    Plots metrics across a range of thresholds.

    args:

    df: a dataframe with the following columns:
    label: the target for each example
    preds: name of the column with the predictions
    """
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    def generate_predictions(df, threshold):
        df["prediction"] = df.apply(
            lambda x: 1 if x[preds_column] > threshold else 0, axis=1
        )
        return df

    plot_data = []
    for threshold in torch.linspace(0, 1, 500):
        df = generate_predictions(df, threshold)
        plot_data.append(
            [
                threshold,
                accuracy_score(df.label, df.prediction),
                precision_score(df.label, df.prediction),
                recall_score(df.label, df.prediction),
                f1_score(df.label, df.prediction),
            ]
        )

    thresholds, accuracies, precisions, recalls, f1s = zip(*plot_data)
    plt.plot(thresholds, accuracies)
    plt.plot(thresholds, precisions)
    plt.plot(thresholds, recalls)
    plt.plot(thresholds, f1s)
    ax, fig = plt.gca(), plt.gcf()
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    legend = ["Accuracy", "Precision", "Recall", "F1 Score"]
    ax.legend(legend)
    ax.set_title(title)
    plt.show()
    return plot_data, legend


### Location and GeoPandas ###
def plot_locations(
    df,
    feature="project",
    title="Tag Locations",
    num_features=10,
    forced_features: list = [None],
):
    """
    Plot points from a DataFrame on a map of Canada with a colour legend.
    The points will be coloured based on the top n unique values in the features column.

    Parameters:
    - df: DataFrame or GeoDataFrame with 'latitude' and 'longitude' and other feature columns.
    - features: the name of the column used to colour the points.
    - title: the title of plot
    - num_classes: the number of unique values in the features column to plot.
    - forced_classes: additional classes to include in the plot regardless of their count in the features column.
    """
    # Convert DataFrame to GeoDataFrame if needed
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude)
        )

    # Load Canada map
    canada = gpd.read_file(
        "../../references/map//ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
    ).query("SOVEREIGNT == 'Canada'")

    # Depricated
    # canada = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")).query(
    #     "name == 'Canada'"
    # )

    # Determine feature labels for coloring
    top_features = df[feature].value_counts().nlargest(num_features).index.to_list()
    all_features = ["Other"] + top_features + forced_features
    df["_color"] = df[feature].where(df[feature].isin(all_features), "Other")

    # Create colormap
    # color_map = plt.cm.get_cmap("tab20", len(all_features))
    color_map = plt.colormaps.get_cmap("tab20")

    # Plot base map
    fig, ax = plt.subplots(figsize=(15, 5))
    canada.plot(color="lightgrey", ax=ax)

    # Plot points with legend
    for i, feature_label in enumerate(all_features):
        color = (
            "grey"
            if feature_label == "Other" and not feature_label in forced_features
            else color_map(i)
        )
        points = df[df["_color"] == feature_label]
        points.plot(
            ax=ax, marker="o", color=color, label=feature_label, markersize=5, alpha=0.5
        )

    # Set aspect to equal for maintaining scale
    ax.set_aspect("equal", adjustable="datalim")

    # Customize and show plot
    ax.set_title(title)
    ax.legend(title=feature.capitalize(), markerscale=2)
    plt.show()

    # Example usage
    # plot_locations(df, feature='species_code', num_features=10, forced_features=['OSFL'])
    # plot_locations(df_lite, feature='project', num_features=10, forced_features=['CWS-Ontario Birds of James Bay Lowlands 2021'])


# Take a sample of recordings from a dataframe
def take_sample(df, sample_fraction=0.1, seed=None):
    """
    Take a random sample of recording locations from a dataframe.
    """
    random.seed(seed)
    unique_recordings = list(set(df.recording_id))
    sample_size = round(sample_fraction * len(unique_recordings))
    sample_recordings = random.sample(unique_recordings, sample_size)
    df_sample = df[df.recording_id.isin(sample_recordings)]
    print(
        f"sampled {len(sample_recordings)} recordings from the original {len(unique_recordings)} "
    )
    return df_sample


# train/test split:
def train_test_split_by_location(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Split the dataframe into train and test sets by location.
    """
    locations = df.location_id.unique()
    train_locations, test_locations = train_test_split(
        locations, test_size=test_size, random_state=random_state
    )
    df_test = df.loc[df.location_id.isin(test_locations)]
    df_train = df.loc[df.location_id.isin(train_locations)]
    return df_train, df_test


# Hashing
def get_hash_from_df(df):
    """
    Convert the DataFrame to a hashable string.
    Take a hash of each row, then concatenate the hashes, and finally hash the concatenated hash.
    """
    df_string_to_hash = "".join(pd.util.hash_pandas_object(df, index=False).astype(str))

    # Use hashlib to create a hash of the entire DataFrame
    df_hash_value = hashlib.sha256(df_string_to_hash.encode()).hexdigest()

    print(df_hash_value)
    return df_hash_value


### Viewing input data
def spec_to_audio(spec_filename, audio_path):
    """
    Utility function to get from a precomputed spectrogram back to the same segment of the audio file.

    The filename of the spectrogram is used.
    Filename format:
    recording-<recording_id>.<file_extension>-<offset>-<end>-.pkl
    Example filename:
    recording-4429.mp3-12.0-15.0-.pkl

    Args:
        spec_filename (str): filename of the spectrogram
        audio_path (str): path to the audio files
    Returns:
        path (str): path to the source audio file
        offset (float): offset in seconds from beginning of the recording
        duration (str): duration of the clip in seconds.
    """
    _, rec_file, offset, end, _ = spec_filename.split("-")
    duration = float(end) - float(offset)
    path = Path(f"{audio_path}/recording-{rec_file}")
    return path, float(offset), duration


def inspect_input_samples(train_df, valid_df, model):
    """
    show a quick sample of present and absent samples in training and validation sets after model preprocessing.
    """
    present_t = train_df.loc[train_df.target_present == 1]
    absent_t = train_df.loc[train_df.target_present == 0]
    present_v = valid_df.loc[valid_df.target_present == 1]
    absent_v = valid_df.loc[valid_df.target_present == 0]

    # Generate a dataset with the samples we wish to inspect and the model's preprocessor
    for df in [present_t, absent_t, present_v, absent_v]:
        inspection_dataset = opso.AudioFileDataset(df.sample(12), model.preprocessor)
        inspection_dataset.bypass_augmentations = True
        samples = [sample.data for sample in inspection_dataset]
        _ = show_tensor_grid(samples, 4, invert=True)


def verify_samples(
    df: pd.DataFrame, ground_truth=1.0, loss_sorted=False, autolabel=False
):
    """
    Present an unverified sample to the user for label verification.
    The dataframe needs a couple of extra columns:
    'loss' which is the absolute difference between a pretrained model's prediction and the ground truth.
    'predicted' which is the model's prediction.
    'confidence_cat' which is the user's confidence in the label. 0=unverified.

    Args:
    df: DataFrame indexed by path, start time , end time. Also needs the columns 'loss', 'predicted', 'confidence_cat', 'target_present'.

    ground_truth: the target class to inspect. 1.0 for present, 0.0 for absent.

    loss_sorted: if True, the clips will be sorted by highest loss first. Otherwise they will be chosen in the order they appear in the dataframe.

    autolabel: if a confidence value is provided, the user will not be prompted for input and can just hit return to autolabel the clip.

    """
    # Filter the split dataset further into unverified and present tags.
    unverified = df[df["confidence_cat"] == 0]
    unverified_target_clips = unverified.loc[
        unverified["target_present"] == ground_truth
    ]
    if len(unverified_target_clips) == 0:
        print("No unverified clips within chosen target class.")
        return df

    if loss_sorted:
        # Sort the unverified clips by the loss value.
        unverified_target_clips = unverified_target_clips.sort_values(
            by="loss", ascending=False
        )

    # Set the confidence cat to 0 so that any skipped clips or crashes don't get saved as previous confidence.
    user_confidence = 0

    clip_idx = unverified_target_clips.index[0]
    path, offset, end_time = clip_idx
    duration = end_time - offset
    audio = opso.Audio.from_file(path, offset=offset, duration=duration)
    spec = opso.Spectrogram.from_audio(audio)
    print(clip_idx)
    print(
        f"target = {df.loc[clip_idx].target_present}, prediction = {df.loc[clip_idx].predicted} loss = {df.loc[clip_idx].loss}"
    )
    audio.show_widget(autoplay=True)

    spec.plot()
    if autolabel:
        label = input("press enter to autolabel")
        if label:
            user_confidence = label
        else:
            user_confidence = autolabel

    elif not autolabel:
        user_confidence = input(
            "enter confidence: 1=Discard, 2=Unsure, 3=Verified, 4=Focal, 5=Re-label-as-present, 6=Re-label-as-absent"
        )

    # save the user input confidence back to the original dataframe.
    df.loc[clip_idx, "confidence_cat"] = int(user_confidence)

    # display the counts
    print(f"added confidence tag {user_confidence} to the dataframe.")
    print(f"verification counts for target_present = {ground_truth}")
    print(df.loc[df.target_present == ground_truth].confidence_cat.value_counts())

    return df


### Evaluation
def get_binary_targets_scores(
    target_df: pd.DataFrame, model_predictions_df: pd.DataFrame, threshold=0.5
):
    """
    Calculate the binary predictions needed for confusion matrix and other metrics.
    target_df: DataFrame with labels in the target_presence column
    model_predictions_df: DataFrame with model predictions in target_presence column
    Returns:
    binary_preds: binary predictions as 0 or 1
    targets: true labels as 0 or 1
    scores: model scores as continuous variables
    """
    targets = target_df.target_presence.values
    scores = model_predictions_df.target_presence.values
    binary_preds = (scores > threshold).astype(float)
    return binary_preds, targets, scores


### Error checking ###
def get_recording_durations(df):
    durations = []
    for idx in tqdm(df.index, desc="getting_audio_file_durations"):
        durations.append(opso.Audio.from_file(idx[0]).duration)
    return durations


def remove_short_clips(df):
    """
    Removes samples from the index of a dataframe that are reported as being short during training.

    This is done by extracting the index information from the file short_samples.log and removing the corresponding samples from the dataframe. First the short_samples.log in the current directory needs
    """

    # read the contents of invalid_samples.log
    with open("short_samples.log") as f:
        short_samples = f.readlines()
    lines = [x.strip().strip("Path: ") for x in short_samples]
    lines = [x.split(",") for x in lines]
    paths = [Path(x[0]) for x in lines]
    starts = [(x[1]).strip(" start_time: ").strip(" sec") for x in lines]
    starts = [float(x) for x in starts]
    ends = [(x[2]).strip(" end_time ").strip(" sec.") for x in lines]
    ends = [float(x) for x in ends]
    short_clips = list(zip(paths, starts, ends))
    short_clips_in_df = df.loc[df.index.isin(short_clips)]
    if len(short_clips_in_df) > 0:
        print(f"{len(short_clips_in_df)} short clips dropped from the dataframe:")
        df.drop(short_clips_in_df.index, inplace=True)
    else:
        print("No short clips found in the dataframe.")
    return df
