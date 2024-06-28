# Installation on Windows

1. Install WSL:<br> 
Follow the instructions at http://opensoundscape.org/en/latest/installation/windows.html in order to install Windows Subsysyem for Linux, but when prompted to install opensoundscape __0.10.1__, install opensoundscape __0.10.0__ instead.

2. In addition to the instructions on opso, you may need to use the following powershell commands:

`wsl.exe --install --no-distribution`

`wsl --list --online`

`wsl --install --distribution Ubuntu-22.04`


3. Create and activate a conda environment<br>
Enter the following in the teminal if you haven't already done so:

`conda create --name osfl-recognizer python=3.10`

`conda activate osfl-recognizer`

`pip install numpy<2.0`

`pip install opensoundscape==0.10.0`

4. Clone this github repository from within Windows Subsystem for Linux, or download it as a zip file and unzip it in
Linux / Ubuntu.22.04 / home / "your user name" / osfl_cnn_recognizer

5. Download OSFL.model [here](https://www.dropbox.com/scl/fi/cx2rblf6yyyoe19kzm4um/OSFL.model?rlkey=wv7c9ll7n2ie1hdn5rk0m9lox&st=2fjauncs&dl=0) and place it in osfl_cnn_recognizer/models 

This model was trained using opensoundscape 0.10.0 so make sure that's the version you have installed.


# To make predictions

In the WSL terminal, type:

python3 predict.py

Paste or type the relative or absolute path to the recordings you'd like to process.

Enter the number of cpu cores to use for preprocessing. For small jobs use 0, for large jobs use 4 or 8 if your computer has that many cores. 

Try to chunk the work into batches in case something crashes, since currently saving only happens on completion of the predict.py program. 

This model will make predictions on all of the wav, flac or mp3 files in the child folders of the provided directory. 

The results will be saved in a .csv file called OSFL-Scores-<current-time>.csv at the project root. 

Predictions are made as a 3 second window moves along with a 1.5 second hop. 

Predictions are stored for each window along the recording, as a new entry in the csv file.


## Installation on mac and linux
This is the same as installation on windows, except that you don't need to install WSL. 
Follow the instructions on how to install opensoundscape with Anaconda here, but use opensoundscape 0.10.0 instead:
http://opensoundscape.org/en/latest/installation/mac_and_linux.html#installation-via-anaconda



