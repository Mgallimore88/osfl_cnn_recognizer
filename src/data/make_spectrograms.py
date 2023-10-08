from pathlib import Path
from opensoundscape.audio import Audio, Spectrogram
import pandas as pd

# Load the processed DataFrame
data_path = Path("../../data/")
df = pd.read_csv("data/processed/processed.csv")

# set paths
raw_call_audio_path = Path.joinpath(data_path, "raw", "call", "audio")
raw_nocall_audio_path = Path.joinpath(data_path, "raw", "nocall", "audio")
call_save_path = Path.joinpath(data_path, "processed", "call", "spectrograms")
nocall_save_path = Path.joinpath(data_path, "processed", "nocall", "spectrograms")
