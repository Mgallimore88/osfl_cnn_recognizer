from pathlib import Path
from opensoundscape import Audio, Spectrogram
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import re


# Load the processed DataFrame
data_path = Path("../../data/")
df = pd.read_pickle(Path.joinpath(data_path, "interim", "processed_metadata.pkl"))


# set paths
raw_call_audio_path = Path.joinpath(data_path, "raw", "call", "audio")
raw_nocall_audio_path = Path.joinpath(data_path, "raw", "nocall", "audio")
processed_call_audio_path = Path.joinpath(data_path, "processed", "call", "audio")
processed_nocall_audio_path = Path.joinpath(data_path, "processed", "nocall", "audio")
call_save_path = Path.joinpath(data_path, "processed", "call", "spectrograms")
nocall_save_path = Path.joinpath(data_path, "processed", "nocall", "spectrograms")

paths = [
    processed_nocall_audio_path,
    processed_call_audio_path,
    call_save_path,
    nocall_save_path,
]
for path in paths:
    if not path.exists():
        path.mkdir(parents=True)

raw_call_audio_files = glob(str(raw_call_audio_path) + "/*")
raw_nocall_audio_files = glob(str(raw_nocall_audio_path) + "/*")


def show_spec_from_audio(file_path):
    audio = Audio.from_file(file_path)
    spec = Spectrogram.from_audio(audio)
    image = spec.to_image(shape=image_shape, invert=True)
    return image


# Trim a short segment to 3 seconds in length or loop the sample to 3 seconds.
def resize_clip(audio_file, clip_length):
    audio = Audio.from_file(audio_file)

    if audio.duration < clip_length:
        audio = audio.loop(clip_length)
    elif audio.duration > clip_length:
        audio = audio.extend_to(clip_length)
    elif audio.duration == clip_length:
        pass

    return audio


def make_uniform_spectrogram(audio_file, clip_length, image_shape):
    # make spectrograms from equal duratiuon clips
    clip = resize_clip(audio_file, clip_length)
    spec = Spectrogram.from_audio(clip)
    image = spec.to_image(shape=image_shape, invert=True)
    return image


# save multiple spectrograms to disk
def make_and_save_specs(audio_files, clip_length, image_shape, image_save_path):
    exceptions = 0
    skipped = 0
    for audio_file in audio_files:
        try:
            image = make_uniform_spectrogram(audio_file, clip_length, image_shape)
        except:
            print(f"could not read file {audio_file}")
            exceptions += 1
        else:
            # save the spectrogram if it hasn't been saved already.
            fname = re.sub("\.[\w]+", ".png", (Path(audio_file).name))
            if Path.exists(image_save_path.joinpath(fname)):
                skipped += 1
            else:
                image.save(image_save_path.joinpath(fname), format="png")

    print(
        f"finished making spectrograms. Skipped {skipped} with {exceptions} exceptions"
    )


clip_length = 3.0
image_shape = (224, 224)

make_and_save_specs(
    raw_call_audio_files[:1000], clip_length, image_shape, call_save_path
)
make_and_save_specs(
    raw_nocall_audio_files[:1000], clip_length, image_shape, nocall_save_path
)
