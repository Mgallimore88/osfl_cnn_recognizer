# script to take a cleaned DataFrame as input and output a set of audio files.
# A sample of 1000 olive sided flycatcher recordings is used for speeding up early development.

import pandas as pd
from pathlib import Path
import requests

def choose_osfl_clips(df, n=1000):
    osfl_idxs = df[df.species_code == "OSFL"].index
    osfls = df.loc[osfl_idxs]
    return df.sample(n, random_state=42)

if __name__ == "__main__":

    data_path = Path("../../data")

    meta = pd.read_pickle(data_path / "interim/processed_metadata.pkl")
    print(meta.head())

    sample_osfls = choose_osfl_clips(meta)



    def exists(fname):
        """
        check to see whether a file exists
        """
        return Path.exists(fname)


    # Download audio clips
    print(f"downloading {len(sample_osfls)} clips")
    rec_path = Path.joinpath(data_path, "raw", "call", "audio")
    Path.mkdir(rec_path, parents=True, exist_ok=True)
    skipped_files = 0
    for i in sample_osfls.index:
        clip_url = sample_osfls.clip_url[i]
        extension = sample_osfls.file_type[i]
        file = Path.joinpath(
            rec_path, f"recording-{sample_osfls.recording_id[i]}-clip-{i}.{extension}"
        )

        if exists(file):
            skipped_files += 1
        else:
            r = requests.get(clip_url)
            with open(file, "wb") as f:
                f.write(r.content)
    print(f"skipped {skipped_files} previously downloaded files")
