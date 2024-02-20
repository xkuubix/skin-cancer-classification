import os
import typing
import logging
import argparse
import pandas as pd

logger = logging.getLogger(__name__)

def group_df(df: pd.DataFrame):
    # df = df.sort_values(by=['image_id'])
    df = df.groupby('lesion_id')
    df = df.agg({
                    "image_id": list,
                    "dx": 'first',
                    "dx_type": 'first',
                    "age": 'first',
                    "sex": 'first',
                    "localization": list,
                    "dataset": list
                    })
    logger.info("Grouped same ID dataframe cells, len(df) = " +\
                    str(len(df)))
    return df.reset_index()


def ungroup_df(df: pd.DataFrame):
    new_df = pd.concat([pd.DataFrame({
        'lesion_id': [row['lesion_id']] * len(row['image_id']),
        'image_id': row['image_id'],
        'dx': [row['dx']] * len(row['image_id']),
        'dx_type': [row['dx_type']] * len(row['image_id']),
        'age': [row['age']] * len(row['image_id']),
        'sex': [row['sex']] * len(row['image_id']),
        'localization': row['localization'],
        'dataset': row['dataset'],
    }) for _, row in df.iterrows()], ignore_index=True)

    logger.info("Ungrouped dataframe cells by lesion_id")

    return new_df

def insert_paths_df(df: pd.DataFrame,
                    img_path: typing.Union[str, bytes, os.PathLike],
                    seg_path: typing.Union[str, bytes, os.PathLike]):
    logger.info("Inserting (2) paths to dataframe, len(df.columns) = " +\
                 str(len(df.columns)))
    
    df.insert(2, 'img_path', None)
    df.insert(3, 'seg_path', None)
    df['img_path'] = img_path + df['image_id'] + '.jpg'
    df['seg_path'] = seg_path + df['image_id'] + '_segmentation.png'
    
    logger.info("Inserted (2) paths to dataframe, len(df.columns) = " +\
                str(len(df.columns)))
    return df

def get_args_parser(path: typing.Union[str, bytes, os.PathLike]):
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=path,
                        help=help)
    return parser
