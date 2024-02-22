import os
import typing
import logging
import argparse
import pandas as pd

logger = logging.getLogger(__name__)

def group_df(df: pd.DataFrame):
    # df = df.sort_values(by=['image_id'])
    old_len = len(df)
    df = df.groupby('lesion_id')
    df = df.agg({
                    "image_id": list,
                    "img_path": list,
                    "seg_path": list,
                    "dx": 'first',
                    "dx_type": 'first',
                    "age": 'first',
                    "sex": 'first',
                    "localization": list,
                    "dataset": list
                    })
    logger.info("Grouped same ID (lesion_id) dataframe cells, " +\
                f"old length = {old_len} => new length = {str(len(df))}")
    return df.reset_index()


def ungroup_df(df: pd.DataFrame):
    old_len = len(df)
    new_df = pd.concat([pd.DataFrame({
        'lesion_id': [row['lesion_id']] * len(row['image_id']),
        'image_id': row['image_id'],
        'img_path': row['img_path'],
        'seg_path': row['seg_path'],
        'dx': [row['dx']] * len(row['image_id']),
        'dx_type': [row['dx_type']] * len(row['image_id']),
        'age': [row['age']] * len(row['image_id']),
        'sex': [row['sex']] * len(row['image_id']),
        'localization': row['localization'],
        'dataset': row['dataset'],
    }) for _, row in df.iterrows()], ignore_index=True)

    logger.info("Ungrouped dataframe cells by lesion_id, " +\
                f"old length = {old_len} => new length = {str(len(new_df))}")
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

def random_split_df(df: pd.DataFrame,
                    train_rest_frac, val_test_frac,
                    seed) -> tuple:

    # print('\n seed = ', seed)
    train = df.sample(frac=train_rest_frac, random_state=seed)
    x = df.drop(train.index)
    val = x.sample(frac=val_test_frac, random_state=seed)
    test = x.drop(val.index)
    logger.info(f"Splitted data into | train-val-test | {len(train)}-{len(val)}-{len(test)} | " +\
                f"{len(train)/len(df)*100:.2f}%-{len(val)/len(df)*100:.2f}%-{len(test)/len(df)*100:.2f}% |")
    return train, val, test

def get_args_parser(path: typing.Union[str, bytes, os.PathLike]):
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=path,
                        help=help)
    return parser

   
def pretty_print_dict(d):
    #take empty string
    sorted_list = sorted(d.items())
    sorted_dict = {}
    for key, value in sorted_list:
        sorted_dict[key] = value
    pretty_dict = ''  
     
    #get items for dict
    for k, v in sorted_dict.items():
        pretty_dict += f'\n\t{k}:\t{v}'
    #return result
    return pretty_dict
 