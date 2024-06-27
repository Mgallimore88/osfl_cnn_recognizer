### Installation on Windows ###

1. Install WSL: Follow the instructions at http://opensoundscape.org/en/latest/installation/windows.html in order to install Windows Subsysyem for Linux.

2. In addition to the instructions on opso, you may need to use the following powershell commands:

wsl.exe --install --no-distribution

wsl --list --online

wsl --install --distribution Ubuntu-22.04


3. Create and activate a conda environment
If you haven't already, 
enter the following in the teminal:

conda create --name opensoundscape pip python=3.10conda activate opensoundscape
conda activate opensoundscape
pip install opensoundscape==0.10.0

4. clone this github repository from within Windows Subsystem for Linux, or download it as a zip file and unzip it in
Linux / Ubuntu.22.04 / home / <user name> / osfl_cnn_recognizer

5. Download the model OSFL.model
if they aren't included when cloning the repository,
models should be saved in the project root osfl_cnn_recognizer / models 
This model was trained using opensoundscape 0.10.0 so make sure that's the version you have installed.


### Installation on mac and linux ###
This is the same as installation on windows, except that you don't need to install WSL. 
Follow the instructions on how to install opensoundscape with Anaconda here:
http://opensoundscape.org/en/latest/installation/mac_and_linux.html#installation-via-anaconda


### To make predictions ###
This model will make predictions on all of the wav, flac or mp3 files in the child folders of the provided directory. 
The results will be saved in a .csv file called OSFL-Scores-<current-time>.csv at the project root. 
Predictions are made as a 3 second window moves along with a 1.5 second hop. 
Predictions are stored for each window along the recording, as a new entry in the csv file. 

### Usage for prediction ###
In the WSL terminal, type:

python3 predict.py

> paste or type the relative or absolute path to the recordings you'd like to process.
> enter the number of cpu cores to use for preprocessing. For small jobs use 0, for large jobs use 4 or 8 if your computer has that many cores. 
> Try to chunk the work into batches in case something crashes, since currently saving only happens on completion of the predict.py program. 

### To train a new model ###
The packages required for the model training walkthrough notebooks are in requirements.txt. Install these if you intend to run the notebooks. 
In this case, you'll also need to install jupyter notebook. In the terminal type:

sudo apt install jupyter-notebook 
type pip install requirements.txt

