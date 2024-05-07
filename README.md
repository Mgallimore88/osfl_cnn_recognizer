--------
osfl_recognizer
--------

Project: 

This Python project develops a method to build and train a convolutional neural network to recognize 
the presence or absence of a single bird species in an audio recording.  

The project uses the OpenSoundscape library for audio preprocessing and model training.

[OpenSoundscape](http://opensoundscape.org/en/latest/index.html)

This project uses audio collected from locations across Canada, and tagged on WildTrax by human listeners. This project trained a model to detect the Olive Sided flycatcher, species code OSFL.

(WildTrax)[https://wildtrax.ca/]


__To make predictions__ using the model, clone this GitHub repository, install the requirements in requirements.txt, and run python3 predict.py from the terminal. 

The model will process audio files in all the sub-directories of the provided folder, and outputs a .csv file called OSFL-scores.csv which contains the probability of detection for each 3 second segment of the recording files as a moving window advances in 1.5 second increments. 

__To train a new model__ look at notebooks/model_training_walkthrough, which contians a series of notebooks showing the training process from start to finish. Anybody wanting to train a similar model in the future can see the process and will have a good starting point to build upon, however the audio used to train the model needs obtaining from a WildTrax organization. 

The OpenSoundscape documentation and tutorials should be studied and understood since many of the processes used here are implementations of the tools provided by OpenSoundscape. 

Source code is located in the .src folder.


--------
Installation
--------
For manual installation:
from the terminal type
- clone this GitHub repository
- install conda
- open bash terminal then run 
conda create --name osfl pip python=3.10
conda activate osfl
pip install requirements.txt
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


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
