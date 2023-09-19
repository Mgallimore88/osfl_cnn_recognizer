# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Data downloader and cleaner
    import pandas as pd
    import requests
    import re
    from pathlib import Path

    data_path = Path('../../data/raw/')

    #Load the csv file
    meta = pd.read_csv(data_path/'TrainingData_BU&Public_CWS.csv')

    # Clean the data, convert ids to integer types


    meta.drop(index=1152839, inplace=True)
    meta.recording_id = meta.recording_id.astype(int)
    meta.location_id = meta.location_id.astype(int)
    meta.task_id = meta.task_id.astype(int)
    meta.tag_id = meta.tag_id.astype(int)
    meta.abundance = meta.abundance.astype(str)

    # Drop tmtt abundance tags.
    tmtt_idxs = meta[meta.abundance=='TMTT'].index
    meta.drop(tmtt_idxs, inplace=True)

    # Drop non song vocalizations
    not_song_idxs = meta[meta.vocalization!='Song'].index
    meta.drop(not_song_idxs, inplace=True)

    # Drop recordings not labeled in wildtrax
    labeled_elsewhere_idxs = meta[meta.tagged_in_wildtrax=='f'].index
    meta.drop(labeled_elsewhere_idxs, inplace=True)

    # choose OSFL entries
    osfl_idxs = meta[meta.species_code=='OSFL'].index
    osfls = meta.loc[osfl_idxs]

    # Add a column named `file_type` to the dataframe. This is done only to samples with a clip_url.
    osfls['file_type'] = None
    for idx in osfls[~osfls.clip_url.isna()].index:
        
        osfls['file_type'][idx] = osfls['clip_url'][idx].split('.')[-1]

    # drop entries with a missing clip_url field from OSFLs
    osfls.drop(osfls.loc[osfls.clip_url.isna()].index, inplace=True)

    # Download audio clips if they don't already exist in data/raw/recordings

    def exists(fname):
        '''
        check to see whether a file exists
        '''
        return Path.exists(fname)

    rec_path = Path.joinpath(data_path, 'recordings')

    for rec in (osfls.index):
        clip_url = osfls.clip_url[rec]
        ext = osfls.file_type[rec]
        file = Path.joinpath(rec_path, str(osfls.recording_id[rec]) + '.' + ext)
        
        if exists(file):
            print(f'{file} already downloaded')
        else:
            r = requests.get(clip_url)
            with open(file, 'wb') as f:
                f.write(r.content)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

