--------
osfl_recognizer
--------

Project: 

This Python project develops a method to build and train a convolutional neural network to recognize 
the presence or absence of a single bird species in an audio recording.  

The project uses the [OpenSoundscape](http://opensoundscape.org/en/latest/index.html) library for audio preprocessing and model training.


This project uses audio collected from locations across Canada, and tagged on (WildTrax)[https://wildtrax.ca/] by human listeners. This project trained a model to detect the Olive Sided flycatcher, species code OSFL.




## To make predictions using the model
- you'll need [git](https://git-scm.com/) and [Anaconda](https://anaconda.org/) installed. 
- Additionally if you're using Windows, follow the instructions [here](./installing opensoundscape on windows.txt) to install WSL

- clone this GitHub repository by running the following command in the terminal:<br>
`git clone https://github.com/Mgallimore88/osfl_cnn_recognizer.git` <br>
- change directory to the project root:<br>
`cd osfl-cnn-recognizer` <br>
- Download OSFL.model from [here](https://www.dropbox.com/scl/fi/cx2rblf6yyyoe19kzm4um/OSFL.model?rlkey=wv7c9ll7n2ie1hdn5rk0m9lox&st=2fjauncs&dl=0) and place it in osfl_cnn_recognizer/models <br>

- create and activate a new conda environment with python 3.9 or higher by running the following commands in a bash terminal: <br>
`conda create --name osfl-recognizer python==3.9` <br>
`conda activate osfl-recognizer` <br>
`pip install -r requirements.txt` <br>

__making predictions__
`python3 predict.py` <br>
Enter the absolute path to the folder containing the recordings you want to analyze when prompted.

The model will process audio files in all the sub-directories of the provided folder, and outputs a .csv file called OSFL-scores.csv which contains the confidence of detection for each 3 second segment of the recording files as a moving window advances in 1.5 second increments. 

Suggested threshold to start ~ 0.8

## To train a new model 
look at _notebooks/model_training_walkthrough_, which contians a series of notebooks showing the training process from start to finish. Anybody wanting to train a similar model in the future can see the process and will have a good starting point to build upon, however the audio used to train the model needs obtaining from a WildTrax organization. 

The OpenSoundscape documentation and tutorials should be studied and understood since many of the processes used here are implementations of the tools provided by the OpenSoundscape library.

Source code is located in the .src folder, and is intended to be used in the context of the notebooks.


--------
Installation
--------
For manual installation:
from the terminal type
- clone this GitHub repository
- open bash terminal then run 
conda create --name osfl pip python=3.10
conda activate osfl
pip install opensoundscape==0.10.0

more details in 'installing opensoundscape.txt'
--------
Usage:
--------

python3 predict.py

--------
Project Organization
--------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │    │
    ├── models             <- Trained models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data - to be used in 
    |                         the context of the walkthrough notebooks. 
    │   
    ├── utils.py           <- Where the custom functions used during model
                              development are stored.
    

The initial exploratory notebooks are removed from the main branch of this project, and can be found on the dev branch.