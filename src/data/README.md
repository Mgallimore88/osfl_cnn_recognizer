Data cleaning and processing I/O and scripts:

TrainingData_BU&Public_CWS_with_rec_links.csv -> clean_csv.py
-
clean_csv.py -> processed_metadata.pkl
-
processed_metadata.pkl -> train_test_split.py -> train_set.pkl
                                              ->  test_set.pkl


train_set.pkl -> /notebooks/exploratory_nbs/1.14-mjg-osfl-check-build-dataset.ipynb
