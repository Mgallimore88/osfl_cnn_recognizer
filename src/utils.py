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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import subprocess
import warnings
import wandb


### General ###
def get_current_git_branch():
    # Run the git command to get the current branch
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return None


def suppress_warnings_if_main_branch():
    current_branch = get_current_git_branch()
    if current_branch == "main":  # Check if the branch is 'main'
        warnings.filterwarnings("ignore")  # Suppress all warnings


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


def clean_confidence_cats(df, drop_unchecked=False):
    """
    drops confidence = 2 or 1, and corrects the labels for confidence = 5 and 6.
    confidence cat 5 means the target is present, 6 means the target is absent.
    """
    # Re-label the mis-labelled clips
    df.loc[df["confidence_cat"] == 5, "target_present"] = 1.0
    df.loc[df["confidence_cat"] == 6, "target_present"] = 0.0

    # drop the clips with confidence 1 or 2 since these were hard to label and wouldn't constitute clear examples of the target class.
    df = df[df["confidence_cat"] != 1]
    df = df[df["confidence_cat"] != 2]
    if drop_unchecked:
        # Drop the unverified clips
        df = df[df["confidence_cat"] != 0]
    return df


def add_missing_metadata_to_df(
    new_df: pd.DataFrame, source_df: pd.DataFrame, columns_to_add: list
):
    """
    Fills in missing metadata to the new dataframe.
    Metadata is taken from the first matching row in the full dataframe.
    This is done at the file level.
    It should only be used for columns which are the same for all clips in a file.
    """
    # Initialize latitude and longitude columns in df_focal
    add_columns = columns_to_add
    for column in add_columns:
        new_df[column] = None

    # Loop through each row in new_df to assign latitude and longitude
    for idx, row in new_df.iterrows():
        # Fetch the first matching row in source_df for the same 'file'
        new_columns = source_df.loc[idx[0]].iloc[0][add_columns]

        # Assign new columns
        for column in add_columns:
            new_df.at[idx, column] = new_columns[column]


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
    df,
    preds_column: str = "present_pred",
    label_column: str = "target_present",
    title: str = "Metrics across thresholds",
):
    """
    Plots metrics across a range of thresholds.

    args:

    df: a dataframe containing the following:
    label_column: the name of the column containing ground truth for each example
    preds_column: name of the column with the predictions
    """
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    def generate_predictions(df, threshold):
        df["prediction"] = df.apply(
            lambda x: 1 if x[preds_column] > threshold else 0, axis=1
        )
        return df

    plot_data = []
    thresholds = torch.linspace(0, 1, 500)

    for threshold in thresholds:
        df = generate_predictions(df, threshold)
        plot_data.append(
            [
                threshold,
                accuracy_score(df[label_column], df.prediction),
                precision_score(df[label_column], df.prediction),
                recall_score(df[label_column], df.prediction),
                f1_score(df[label_column], df.prediction),
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


def inspect_input_samples(train_df, valid_df, model, bypass_augmentations=True):
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
        inspection_dataset.bypass_augmentations = bypass_augmentations
        samples = [sample.data for sample in inspection_dataset]
        _ = show_tensor_grid(samples, 4, invert=True)


def show_samples_in_df(df, model):
    """
    show a quick sample of all the samples in a dataframe after model preprocessing.
    """
    # Generate a dataset with the samples we wish to inspect and the model's preprocessor
    inspection_dataset = opso.AudioFileDataset(df, model.preprocessor)
    inspection_dataset.bypass_augmentations = True
    samples = [sample.data for sample in inspection_dataset]
    _ = show_tensor_grid(samples, 2, invert=True)


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
    norm_spec = opso.Spectrogram.from_audio(audio.normalize())
    print(clip_idx)
    print(
        f"target = {df.loc[clip_idx].target_present}, prediction = {df.loc[clip_idx].predicted} loss = {df.loc[clip_idx].loss}"
    )
    audio.show_widget(autoplay=True)

    spec.plot()
    norm_spec.plot()
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
    target_df: DataFrame with labels in the target_present column
    model_predictions_df: DataFrame with model predictions in target_present column
    Returns:
    binary_preds: binary predictions as 0 or 1
    targets: true labels as 0 or 1
    scores: model scores as continuous variables
    """
    targets = target_df.target_present.values
    scores = model_predictions_df.target_present.values
    binary_preds = (scores > threshold).astype(float)
    return binary_preds, targets, scores


def plot_confusion_matrix(
    target_df: pd.DataFrame,
    model_predictions_df: pd.DataFrame,
    threshold=0.5,
    title: str = "Confusion Matrix",
):
    """
    Plot a confusion matrix.
    The inputs should be two different pandas dataframes - one containing the target labels and the other containing the model predictions.
    The dataframe columns should both be named "target_present" for simplicity when viewing predictions made using opensoundscape.
    """
    preds, targets, scores = get_binary_targets_scores(
        target_df, model_predictions_df, threshold
    )
    # plot confusion matrix
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm)
    ax = disp.plot(colorbar=False).ax_
    ax.set_title(f"{title} \n threshold: {threshold}")
    return cm


### WandB ###
def log_single_metric_to_wandb(metric, thresholds, name):
    data = [[x, y] for (x, y) in zip(thresholds, metric)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log(
        {
            f"custom {name}": wandb.plot.line(
                table, "x", "y", title=f"Custom {name} vs threshold plot"
            )
        }
    )


def hawkears_files_to_df(hawkears_output_files, target_species="OSFL"):
    """
    Takes a list of file paths containing hawkears predictions as input.
    Predictions should be made on 3 second audio files saved to disk.
    Returns a dataframe of predictions per file.
    This assumes the format of the hawkears output files is as follows:
    start_time	end_time	species_code;confidence
    """
    results = []
    # Read and parse the file contents
    for file_path in hawkears_output_files:
        with open(file_path, "r") as f:
            file_content = f.read().strip().split("\n")
            file_id = file_path.split("/")[-1].split("-")[0]
            file_id = file_id.split("_")[0]
            for detection in file_content:
                if detection:
                    line = [file_id] + detection.split("\t")
                else:
                    line = [file_id]
                results.append(line)

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(
        results, columns=["file_id", "start_time", "end_time", "species;confidence"]
    )
    df.file_id = df.file_id.astype(int)
    df.sort_values(by=["file_id"], inplace=True)
    df[["species", "confidence"]] = df["species;confidence"].str.split(";", expand=True)
    df.drop(columns=["species;confidence"], inplace=True)
    df = df.loc[df["species"] == target_species]
    df.confidence = df.confidence.astype(float)
    return df.reset_index(drop=True)


def save_dataframe_clips_to_disk(df: pd.DataFrame, save_path: Path):
    """
    function for saving the 3 second clips which make up a dataset to disk.
    saved filename contains a numeric index and the file extension.
    args:
    df: pandas dataframe with column ['file_ID'] containing unique numeric index per clip
    save_path: path to save the clips to
    """
    if not any(save_path.iterdir()):
        print("Saving clips to disk")
        i = 0
        for index in df.index:
            path, start, end = index
            row_id = int(df.iloc[i].file_ID)
            clip = audio.Audio.from_file(path, offset=start, duration=end - start)
            extension = str(path).split(".")[-1]
            clip.save(save_path / f"{str(row_id)}.{extension}", suppress_warnings=True)
            i += 1
    else:
        print(
            "Directory is not empty. Set an empty save directory before saving clips."
        )


### Error checking ###
def get_recording_durations(df):
    durations = []
    for idx in tqdm(df.index, desc="getting_audio_file_durations"):
        durations.append(opso.Audio.from_file(idx[0]).duration)
    return durations


def remove_short_clips(df):
    """
    Removes samples from the index of a dataframe that are reported as being short during training.

    This is done by extracting the index information from the file short_samples.log and removing the corresponding samples from the dataframe. The paths to the samples are saved in short_sample.log in the following format:

    Path: ../../data/raw/recordings/OSFL/recording-207234.mp3, start_time: 340.5 sec, end_time 343.5 sec.

    """

    # read the contents of invalid_samples.log
    with open("short_samples.log") as f:
        short_samples = f.readlines()

    # extract the index from the log
    lines = [x.strip().strip("Path: ") for x in short_samples]
    lines = [x.split(",") for x in lines]
    paths = [Path(x[0]) for x in lines]
    starts = [(x[1]).strip(" start_time: ").strip(" sec") for x in lines]
    starts = [float(x) for x in starts]
    ends = [(x[2]).strip(" end_time ").strip(" sec.") for x in lines]
    ends = [float(x) for x in ends]

    # make a list of the indices to drop, and drop them if there are any.
    short_clips = list(zip(paths, starts, ends))
    short_clips_in_df = df.loc[df.index.isin(short_clips)]
    if len(short_clips_in_df) > 0:
        print(f"{len(short_clips_in_df)} short clips dropped from the dataframe:")
        df.drop(short_clips_in_df.index, inplace=True)
    else:
        print("No short clips found in the dataframe.")
    return df
