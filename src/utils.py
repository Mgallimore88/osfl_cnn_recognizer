import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from IPython.display import display


### Pandas ###
def display_all(df, max_rows=None, max_columns=None):
    """
    Display all columns and rows of a dataframe without truncating.
    """
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
]


### Graphing ###
def print_stats(df):
    """
    simple stats
    """
    return df.describe()


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
    # plot_locations(df_lite, feature='species_code', num_features=10, forced_features=['OSFL'])
    # plot_locations(df_lite, feature='project', num_features=10, forced_features=['CWS-Ontario Birds of James Bay Lowlands 2021'])
