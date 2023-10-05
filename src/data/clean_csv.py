# Preprocessing script to load raw csv file and create a modified, cleaned DataFrame as output.


import pandas as pd
import re
from pathlib import Path
from preset_types import type_dict

data_path = Path("../../data/")

# Load the raw csv file
meta = pd.read_csv(data_path / "raw/TrainingData_BU&Public_CWS_with_rec_links.csv")

# Drop last entry since it's all NaN values.
meta.drop(meta.tail(1).index, inplace=True)

# Replace empty fields with -1 for
meta.loc[meta["verifier_id"].isna(), "verifier_id"] = -1

# Change all the data types in the DataFrame to the types specified in in the preset_types.py
meta = meta.astype(type_dict)

# Drop 'too many to tag' abundance tags.
tmtt_idxs = meta[meta.abundance == "TMTT"].index
meta.drop(tmtt_idxs, inplace=True)

# Drop non song vocalizations
not_song_idxs = meta[meta.vocalization != "Song"].index
meta.drop(not_song_idxs, inplace=True)

# Drop recordings not labeled in wildtrax
labeled_elsewhere_idxs = meta[meta.tagged_in_wildtrax == "f"].index
meta.drop(labeled_elsewhere_idxs, inplace=True)

# Remove the clips which don't contain a link to a recording
meta.drop(meta.loc[meta.clip_url == "nan"].index, inplace=True)

# Add a column to store file type from clip URL
meta["file_type"] = None
has_url = ~(meta.clip_url == "nan")


def get_file_type(url: str) -> str:
    return url.split(".")[-1]


meta["file_type"] = meta.recording_url.apply(get_file_type)


# Export the cleaned version of the database
interim_data_path = Path("../../data/interim/")
meta.to_csv(interim_data_path / "processed_metadata.csv")
meta.to_pickle(interim_data_path / "processed_metadata.pkl")
