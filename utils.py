import os
import typing
import logging
import argparse
import pandas as pd
import random

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
    """
    Randomly splits a DataFrame into train, validation, and test sets.

    Parameters:
    df (pd.DataFrame): The DataFrame to be split.
    train_rest_frac (float): The fraction of data to be used for training.
    val_test_frac (float): The fraction of remaining data to be used for validation and testing.
    seed (int): The random seed for reproducibility.

    Returns:
    tuple: A tuple containing the train, validation, and test sets.
    """
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

   
def pretty_dict_str(d, key_only=False):
    #take empty string
    sorted_list = sorted(d.items())
    sorted_dict = {}
    for key, value in sorted_list:
        sorted_dict[key] = value
    pretty_dict = ''  
     
    #get items for dict
    if key_only:
        for k, _ in sorted_dict.items():
            pretty_dict += f'\n\t{k}'
    else:
        for k, v in sorted_dict.items():
            pretty_dict += f'\n\t{k}:\t{v}'
        #return result
    return pretty_dict
 
def oversample_data(data, key_to_use='dx'):
    """
    Oversamples the data based on the specified key.

    Args:
        data (list of dicts): The input data to be oversampled.
        key_to_use (str, optional): The key to use for oversampling. Defaults to 'dx'.

    Returns:
        list of dicts: The oversampled data.

    """
    k_values = [row[key_to_use] for row in data]
    k_counts = {k: k_values.count(k) for k in set(k_values)}

    # total_instances = len(data)
    instances_per_class = max(k_counts.values())
    oversampling_ratio = {k: instances_per_class / count for k, count in k_counts.items()}

    logger.info("Prior number of key matching rows:" +
                pretty_dict_str(k_counts) +
                f"\n" + "_"*20 +
                f"\nTotal instances:{sum(k_counts.values())}")

    oversampled_data = []
    for row in data:
        k_value = row[key_to_use]
        oversampled_data.extend([row] * int(oversampling_ratio[k_value]))
    
    k_values = [row[key_to_use] for row in oversampled_data]
    k_counts = {k: k_values.count(k) for k in set(k_values)}
    logger.info("Number of rows after oversampling:\n" +
                pretty_dict_str(k_counts) +
                f"\n" + "_"*20 +
                f"\nTotal instances:{sum(k_counts.values())}")

    return oversampled_data

def undersample_data(data, key_to_use='dx', seed=42, multiplier=1.0):
    """
    Undersamples the given data based on a specified key.

    Args:
        data (list): The input data to be undersampled.
        key_to_use (str, optional): The key to use for undersampling. Defaults to 'dx'.
        seed (int, optional): The seed value for randomization. Defaults to 42.
        multiplier (float, optional): The multiplier [0. ... 1.] to determine the minimum count for undersampling. Defaults to 1.0.

    Returns:
        list: The undersampled data.

    Raises:
        AssertionError: If min_count is not greater than 0 or if min_count is greater than the smallest class count.

    """
    random.seed(seed)

    k_values = [row[key_to_use] for row in data]
    k_counts = {k: k_values.count(k) for k in set(k_values)}
    logger.info("Prior number of key matching rows:" +
                pretty_dict_str(k_counts) +
                f"\n" + "_"*20 +
                f"\nTotal instances:{sum(k_counts.values())}")
    
    min_count = int(min(k_counts.values()) * multiplier)
    
    assert min_count > 0, "min_count must be greater than 0"
    assert min_count <= min(k_counts.values()), "min_count must be less than or equal to the smallest class count"

    undersampled_data = []
    for k_value in k_counts:
        # Select random subset of rows for undersampling
        subset = [row for row in data if row[key_to_use] == k_value]
        random.shuffle(subset)
        subset = subset[:min_count]
        undersampled_data.extend(subset)
    k_values = [row[key_to_use] for row in undersampled_data]
    k_counts = {k: k_values.count(k) for k in set(k_values)}
    logger.info("Number of rows after undersampling:\n" +
                pretty_dict_str(k_counts) +
                f"\n" + "_"*20 +
                f"\nTotal instances:{sum(k_counts.values())}")

    return undersampled_data