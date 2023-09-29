# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

# from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Data downloader and cleaner
    import pandas as pd
    import requests
    import re
    from pathlib import Path
    from preset_types import type_dict

    data_path = Path("../../data/raw/")

    # Load the csv file
    meta = pd.read_csv(data_path / "TrainingData_BU&Public_CWS.csv")

    # create a column named 'clip_id' to keep track of original indices.
    meta["clip_id"] = meta.index
    cols = list(meta.columns)
    meta = meta[cols[-1:] + cols[:-1]]

    # Drop last entry since it's all NaN values. 
    meta.drop(meta.tail(1).index, inplace=True)

    # Replace empty fields with -1 
    meta.loc[meta['verifier_user_id'].isna(), 'verifier_user_id'] = -1

    #Change all the data types in the dataframe to the chosen types contained in the type dict.
    meta = meta.astype(type_dict)

    # Drop tmtt abundance tags.
    tmtt_idxs = meta[meta.abundance == "TMTT"].index
    meta.drop(tmtt_idxs, inplace=True)

    # Drop non song vocalizations
    not_song_idxs = meta[meta.vocalization != "Song"].index
    meta.drop(not_song_idxs, inplace=True)

    # Drop recordings not labeled in wildtrax
    labeled_elsewhere_idxs = meta[meta.tagged_in_wildtrax == "f"].index
    meta.drop(labeled_elsewhere_idxs, inplace=True)

    # Remove the clips which don't contain a link to a recording
    meta.drop(meta.loc[meta.clip_url=='nan'].index, inplace=True)

    # Add a column named `file_type` to the dataframe. This is done only to samples with a clip_url.
    meta["file_type"] = None
    for idx in meta[~meta.clip_url.isna()].index:
        meta["file_type"][idx] = meta["clip_url"][idx].split(".")[-1]

    # # choose OSFL entries
    # osfl_idxs = meta[meta.species_code == "OSFL"].index
    # osfls = meta.loc[osfl_idxs]


    # # drop entries with a missing clip_url field from OSFLs
    # osfls.drop(osfls.loc[osfls.clip_url.isna()].index, inplace=True)

    # # Download audio clips if they don't already exist in data/raw/recordings

    # def exists(fname):
    #     """
    #     check to see whether a file exists
    #     """
    #     return Path.exists(fname)



    # Export the cleaned version of the database
    processed_data_path = Path('../../data/processed/')
    meta.to_csv(processed_data_path/'processed_metadata.csv')
    meta.to_pickle(processed_data_path/'processed_metadata.pkl')


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
