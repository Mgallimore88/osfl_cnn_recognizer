import matplotlib.pyplot as plt
import geopandas as gpd


### Location and GeoPandas ###
def plot_locations(df, title="clip_locations"):
    """
    Take a dataframe and plot the lattitude and longitude columns on a map of Canada.
    """
    # initialize an axis
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot map on axis
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    countries[countries["name"] == "Canada"].plot(color="lightgrey", ax=ax)
    # plot points
    df.plot(
        x="longitude",
        y="latitude",
        kind="scatter",
        s=40,
        alpha=0.01,
        c="red",
        title=title,
        ax=ax,
    )
    # add grid
    ax.grid(alpha=0.5)
    plt.show()
