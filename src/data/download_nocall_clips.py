# Take a cleaned DataFrame as input and output a set of audio files.
# A sample of 1000 recordings is taken from samples tagged as species other than olive sided flycatcher.

# IMPORTANT: This method won't guarantee absence of olive sided flycatcher calls in the recordings,
# and is only included as a placeholder for the developing the workflow.

import pandas as pd
from pathlib import Path
import requests

data_path = Path("../../data")

meta = pd.read_pickle(data_path / "interim/processed_metadata.pkl")
print(meta.head())

# choose non OSFL clips
# choose non OSFL clips
nocall_idxs = meta[~(meta.species_code == "OSFL")].index
nocalls = meta.loc[nocall_idxs]
sample_nocalls = nocalls.sample(1000, random_state=42)


def exists(fname):
    """
    check to see whether a file exists
    """
    return Path.exists(fname)


# Download audio clips
print(f"downloading {len(sample_nocalls)} clips")
rec_path = Path.joinpath(data_path, "raw", "nocall", "audio")
Path.mkdir(rec_path, parents=True, exist_ok=True)
skipped_files = 0
for i in sample_nocalls.index:
    clip_url = sample_nocalls.clip_url[i]
    extension = sample_nocalls.file_type[i]
    file = Path.joinpath(
        rec_path, f"recording-{sample_nocalls.recording_id[i]}-clip-{i}.{extension}"
    )

    if exists(file):
        skipped_files += 1
    else:
        r = requests.get(clip_url)
        with open(file, "wb") as f:
            f.write(r.content)
print(f"skipped {skipped_files} previously downloaded files")
