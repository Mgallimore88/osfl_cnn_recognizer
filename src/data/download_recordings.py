import pandas as pd
from pathlib import Path
import requests


def from_url(
    df: pd.DataFrame,
    url_col: str,
    output_path: Path,
    target: str | None = None,
    n: int | None = None,
) -> None:
    """
    Takes a DataFrame and URL column as input and downloads a set of audio recordings.

    Parameters:
    - df: DataFrame with a URL column
    - url_col: name of the URL column
    - output_path: path to save the downloaded files
    - target: species code to filter for
    - n: number of samples to download
    """

    # filter for target species if specified
    if target is not None:
        df = df.loc[df.species_code == target]

    # Get the unique recording URLs so that duplicates aren't downloaded
    df = df.drop_duplicates(subset=[url_col])

    # choose n samples if specified
    if n is not None:
        df = df.sample(n, random_state=42)

    # Download audio clips
    print(f"downloading {len(df)} clips")
    Path.mkdir(output_path, parents=True, exist_ok=True)
    skipped_files = 0
    for i in df.index:
        filename = i  # for clarity - the index is also the filename
        recording_url = df.loc[i, url_col]
        file = Path.joinpath(output_path, filename)

        if Path.exists(file):
            skipped_files += 1
        else:
            print(recording_url)
            r = requests.get(str(recording_url))
            with open(file, "wb") as f:
                f.write(r.content)
    print(f"skipped {skipped_files} previously downloaded files")
