# imports
import opensoundscape
import sys
from pathlib import Path
from datetime import datetime

BASE_PATH = Path.cwd()
sys.path.append(str(BASE_PATH))
data_path = BASE_PATH / "data"
model_save_path = BASE_PATH / "models"

# get the input path from user input
audio_files_path = input("Enter the path to the audio files: ")

audio_files = []
for file_path in Path(audio_files_path).rglob("*"):
    if file_path.suffix in [".wav", ".mp3", ".flac"]:
        absolute_path = file_path
        audio_files.append(absolute_path)

#  make predicitons using the latest model
model = opensoundscape.load_model(model_save_path / "OSFL.model")

cpu_cores = input(
    "Enter the number of CPU cores to use for parallel preprocessing: default is 0: "
)
print("====== Making predictions on audio files...======")
# Make sure you're using the GPU - it should say 'cuda' if you are on a pc and 'mps' on a mac.
print(f"model.device is: {model.device}")
# make predictions
scores = model.predict(
    audio_files,
    overlap_fraction=0.5,
    num_workers=int(cpu_cores),
    activation_layer="sigmoid",
    invalid_samples_log="errors.log",
)

# save the predictions to a csv file
print(f"====== Saved the predictions to OSFL-scores-{datetime.now()}.csv======")
scores.to_csv(f"OSFL-scores-{datetime.now()}.csv", index=True)
