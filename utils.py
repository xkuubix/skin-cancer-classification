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
    logger.debug("Grouped same ID dataframe cells, len(df) = " +\
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

    return new_df

def get_args_parser(path: typing.Union[str, bytes, os.PathLike]):
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=path,
                        help=help)
    return parser

# img_pth: typing.Union[str, bytes, os.PathLike],
# seg_pth: typing.Union[str, bytes, os.PathLike],
# csv_pth: typing.Union[str, bytes, os.PathLike],

# if os.path.exists(csv_pth):
#     df = pd.read_csv(csv_pth)
# else:
#     raise FileNotFoundError

# logger.debug("Metadata has been loaded, len(df) = " +\
#                 str(len(df)))