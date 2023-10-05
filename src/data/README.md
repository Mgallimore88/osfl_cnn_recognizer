Pipeline for data cleaning and processing:

TrainingData_BU&Public_CWS_with_rec_links.csv -> clean_csv.py
-
clean_csv.py -> processed_metadata.pkl
-
processed_metadata.pkl -> download_call_data.py

download_call_data.py -> data/interim/call/audio/*.flac/mp3

download_nocall_data.py -> data/interim/nocall/audio/*.flac/mp3







