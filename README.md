# osfl_recognizer

## About: 

This Python project develops a method to build and train a convolutional neural network to recognize 
the presence or absence of a single bird species in an audio recording.  

A description of the model training process and evaluation is available [here](https://mgallimore88.github.io/blog/posts/osfl-project-description/B-project-description.html)

The trained model is made available to make predictions on local audio files. 

The project uses the [OpenSoundscape](http://opensoundscape.org/en/latest/index.html) library for audio preprocessing and model training.

This project uses audio collected from locations across Canada, and tagged on [WildTrax](https://wildtrax.ca/) by human listeners. This project trained a model to detect the __Olive Sided flycatcher__, species code OSFL.


## To make predictions using the model
- you'll need [git](https://git-scm.com/) and [Anaconda](https://anaconda.org/) installed. 
- Additionally if you're using Windows, follow the instructions [here](windows-install.md) to install WSL

- clone this GitHub repository by running the following command in the terminal:<br>
`git clone https://github.com/Mgallimore88/osfl_cnn_recognizer.git` <br>
- change directory to the project root:<br>
`cd osfl-cnn-recognizer` <br>
- Download OSFL.model from [here](https://www.dropbox.com/scl/fi/cx2rblf6yyyoe19kzm4um/OSFL.model?rlkey=wv7c9ll7n2ie1hdn5rk0m9lox&st=2fjauncs&dl=0) and place it in __osfl_cnn_recognizer/models__ <br>

- create and activate a new conda environment with python 3.9 or higher by running the following commands in a bash terminal: <br>
`conda create --name osfl-recognizer python==3.9` <br>
`conda activate osfl-recognizer` <br>
`pip install -r requirements.txt` <br>

## To make predictions
`python3 predict.py` <br>

Enter the absolute path to the folder containing the recordings you want to analyze when prompted.

Enter the number of cpu cores to use for preprocessing. For small jobs use 0, for large jobs try using 4 or 8 if your computer has that many cores.

The model will process audio files in all the sub-directories of the provided folder, and outputs a .csv file called OSFL-Scores-<current-time>.csv which contains the confidence of detection for each 3 second segment of the recording files as a moving window advances in 1.5 second increments. 

 Try to chunk the work into batches in case something crashes, since currently saving only happens on completion of the predict.py program. 

Suggested threshold to start ~ 0.8

## To train a new model 
look at __notebooks/model_training_walkthrough__, which contians a series of notebooks showing the training process from start to finish. Anybody wanting to train a similar model in the future can see the process and will have a good starting point to build upon, however the audio used to train the model needs obtaining from a WildTrax organization. 

The OpenSoundscape documentation and tutorials should be studied and understood since many of the processes used here are implementations of the tools provided by the OpenSoundscape library.

Source code is located in the .src folder, and is intended to be used in the context of the notebooks.

-------
### Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │    │
    ├── models             <- Trained models
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
    ├── requirements.txt   <- The requirements file for running the model
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
