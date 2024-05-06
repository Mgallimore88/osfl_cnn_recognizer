--------
osfl_recognizer
--------

Project: 

This Python project develops a method to build and train a convolutional neural network to recognize 
the presence or absence of a single bird species in an audio recording.  The project uses the OpenSoundscape library for audio preprocessing and model training.

[OpenSoundscape](http://opensoundscape.org/en/latest/index.html)

The project trained a model to detect the Olive Sided flycatcher, species code OSFL. The audio to train the model was collected from WildTrax, and tagged in WildTrax. 

(WildTrax)[https://wildtrax.ca/]


__To make predictions__ using the model, clone this GitHub repository, install the requirements in requirements.txt, and run python3 predict.py from the terminal. 

The model will process audio files in all the sub-directories of the provided folder, and outputs a .csv file called OSFL-scores.csv which contains the probability of detection for each 3 second segment of the recording files as a moving window advances in 1.5 second increments. 

__To train a new model__ look at notebooks/model_training_walkthrough, which contians a series of notebooks showing the training process from start to finish Anybody wanting to train a similar model in the future can see the process and will have a good starting point to build upon.
The OpenSoundscape documentation and tutorials should be studied and understood since many of the processes used here are implementations of the tools provided by OpenSoundscape. 

Source code is located in the .src folder.


--------
Installation
--------
For manual installation,
- install conda
- open bash terminal then run 
conda create --name osfl pip python=3.10
conda activate osfl
pip install opensoundscape==0.10.0
conda install -c fastai fastai


--------
Repeating the process developed in this project
1. get access to WildTrax database and download a csv file containing human labelled species tag timestamps
2. preprocess the csv file to remove erroneous, duplicated, anomalous and low quality data.
3. Download a sample of audio files from the 'clip_url' links for a variety of species
4. Convert these to spectrograms using OpenSoundScape
5. Train a quick model using the fastai library and out of the box settings to distinguish between bird species. Use the model to get insight into what causes most confusion / highest loss

6. Seperate out a withheld test set of 20% of the recording locations from the database and keep these isolated from future training.
7. Create a further train/validation split
8. Create a dataset of positive and negative class audio samples
9. train a model using these samples
10. iterate the training process using various combinations of mixup, data augmentation, model architectures and hyperparameters until good performance is acheived as measured on the validation set. 
11. Increase the size of the dataset used to train with.
11. Confirm the model's performance by running on the withheld test set. 
12. Export a model and report performance metrics.


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
