==============================
osfl_recognizer
==============================

cnn for recognizing olive sided flycatcher calls in audio recordings.

For manual installation,
- install conda
- open bash terminal then run 
conda create --name osfl pip python=3.10
conda activate osfl
pip install opensoundscape==0.10.0
conda install -c fastai fastai


==============================
Roadmap for dev
==============================
- train initial model on spectrograms with default settings for everything
- curate a good no-call dataset
- curate sensible test and validation splits
- re-train the model and try different architectures and models pretrained on different datasets
- iterate through development until model performance is improved
- try different data augmentation techniques, spectrogram parameters, and model parameters
- try reinforcing the model with focal recordings from xeno canto (Exclude these from the validation and test set)
- once the process is tuned, train the model on the full dataset and increase the model size. 
- record model performance on the test set
- export the model to PyTorch model format
- apply conversion to ONNX format if needed
- create deployment script for the model with environment and dependencies
- document the process and create a tutorial for using the model for inference


Project Organization
==============================

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
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
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
A note on notebooks. 
The exploratory notebooks are kept as a record of the development process. They aren't intended as documentation or reference. Techniques and code described in earlier notebooks might be superceded by newer approaches in later notebooks. 
--------


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
