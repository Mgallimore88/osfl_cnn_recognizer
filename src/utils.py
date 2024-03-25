import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from IPython.display import display
import random
import opensoundscape as opso
import torch
import hashlib
from tqdm import tqdm


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


def calculate_file_durations(df):
    # Takes multi indexed df with file paths as first index. Returns file durations.
    audio_files = df.index.get_level_values("file").unique().values
    opso.Audio.from_file(audio_files[0]).duration
    durations = []
    for file in audio_files:
        audio = opso.Audio.from_file(file)
        durations.append(audio.duration)
    return durations


def clean_confidence_cats(df):
    # Re-label the mis-labelled clips
    df.loc[df["confidence_cat"] == 5, "target_presence"] = 1.0
    df.loc[df["confidence_cat"] == 6, "target_presence"] = 0.0

    # drop the clips with confidence 1 or 2 since these were hard to label and wouldn't constitute clear examples of the target class.
    df = df[df["confidence_cat"] != 1]
    df = df[df["confidence_cat"] != 2]
    return df


def show_sample_from_df(df: pd.DataFrame, label: str | None = "present"):
    """
    Play audio and plot spectrogram for an item in a dataframe.
    Index must be a multi index of path, offset, end time.
    """
    if label == "present":
        sample = df.loc[df.target_presence == 1].sample()
    elif label == "absent":
        sample = df.loc[df.target_presence == 0].sample()
    else:
        sample = df.sample()
    path, offset, end_time = sample.index[0]
    duration = end_time - offset
    audio = opso.Audio.from_file(path, offset=offset, duration=duration)
    spec = opso.Spectrogram.from_audio(audio)
    audio.show_widget()
    spec.plot()


### Graphing ###
def print_stats(df):
    """
    simple stats
    """
    return df.describe()


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
    canada = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")).query(
        "name == 'Canada'"
    )

    # Determine feature labels for coloring
    top_features = df[feature].value_counts().nlargest(num_features).index.to_list()
    all_features = ["Other"] + top_features + forced_features
    df["_color"] = df[feature].where(df[feature].isin(all_features), "Other")

    # Create colormap
    color_map = plt.cm.get_cmap("tab20", len(all_features))

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


### Error checking ###
def get_recording_durations(df):
    durations = []
    for idx in tqdm(df.index, desc="getting_audio_file_durations"):
        durations.append(opso.Audio.from_file(idx[0]).duration)
    return durations
