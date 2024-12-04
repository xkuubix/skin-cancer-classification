import os
import torch
import random
import typing
import logging
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from sklearn.linear_model import LassoCV
from Dataset import MappingHandler
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def group_df(df: pd.DataFrame):
    # df = df.sort_values(by=['image_id'])
    previous_len = len(df)
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
                f"previous length = {previous_len} => new length = {str(len(df))}")
    return df.reset_index()


def ungroup_df(df: pd.DataFrame):
    previous_length = len(df)
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
                f"previous length = {previous_length} => new length = {str(len(new_df))}")
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


def generate_sampler(target):
    """
    Generate a weighted random sampler based on the target array.

    Parameters:
    - target (numpy.ndarray): The target array containing class labels.

    Returns:
    - sampler (torch.utils.data.sampler.WeightedRandomSampler): The weighted random sampler object.
    """

    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight,
                                    len(samples_weight),
                                    replacement=True)
    return sampler


def prepare_data_for_fold(train_fold, val_fold, test_df, random_state, cv=5):

    feature_columns = train_fold.columns[10:]
    X_train = train_fold[feature_columns].values
    
    scaler = StandardScaler()
    mapping_handler = MappingHandler().mapping
    X_train_scaled = scaler.fit_transform(X_train)
    y_train = [int(mapping_handler[label.upper()]) for label in train_fold['dx'].values]
    
    lasso = LassoCV(cv=cv, random_state=random_state, max_iter=10000, tol=0.001)
    lasso.fit(X_train_scaled, y_train)
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if lasso.coef_[i] != 0]
    print(f"No. of selected features: {len(selected_features)}")

    scaler = StandardScaler()
    train_fold[selected_features] = scaler.fit_transform(train_fold[selected_features].values)
    val_fold[selected_features] = scaler.transform(val_fold[selected_features].values)
    test_df[selected_features] = scaler.transform(test_df[selected_features].values)

    # keep only the selected features and metadata columns
    metadata_columns = train_fold.columns[:10]  # Adjust as necessary for metadata columns
    columns_to_keep = list(metadata_columns) + selected_features

    return train_fold[columns_to_keep], val_fold[columns_to_keep], test_df[columns_to_keep]


def print_metrics(fold_results):
    metrics_summary_macro = {
    'precision': [],
    'recall': [],
    'f1-score': [],
    }
    metrics_summary_weighted = {
    'precision': [],
    'recall': [],
    'f1-score': [],
    }
    accuracy = []
    balanced_accuracy = []
    for fold_result in fold_results:
        for label, metrics in fold_result.items():
            if label == 'macro avg':
                metrics_summary_macro['precision'].append(metrics['precision'])
                metrics_summary_macro['recall'].append(metrics['recall'])
                metrics_summary_macro['f1-score'].append(metrics['f1-score'])
            elif label == 'weighted avg':
                metrics_summary_weighted['precision'].append(metrics['precision'])
                metrics_summary_weighted['recall'].append(metrics['recall'])
                metrics_summary_weighted['f1-score'].append(metrics['f1-score'])
            elif label == 'accuracy':
                accuracy.append(metrics)
            elif label == 'balanced_accuracy':
                balanced_accuracy.append(metrics)

# Convert lists to numpy arrays for easy statistical calculations
    for metric in metrics_summary_weighted:
        metrics_summary_weighted[metric] = np.array(metrics_summary_weighted[metric])
    for metric in metrics_summary_macro:
        metrics_summary_macro[metric] = np.array(metrics_summary_macro[metric])
    accuracy = np.array(accuracy)
    balanced_accuracy = np.array(balanced_accuracy)

# Compute mean and standard deviation for each metric
    digits = 6
    print("\nSummary of performance metrics (weighted avg):")
    for metric, values in metrics_summary_weighted.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric.capitalize():9s}\tMean = {mean:.{digits}f},\tSD = {std:.{digits}f}")

    print("\nSummary of performance metrics (macro avg):")
    for metric, values in metrics_summary_macro.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric.capitalize():9s}\tMean = {mean:.{digits}f},\tSD = {std:.{digits}f}")

    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)
    print(f"\n{'Accuracy':9s}\tMean = {accuracy_mean:.{digits}f},\tSD = {accuracy_std:.{digits}f}")

    balanced_accuracy_mean = np.mean(balanced_accuracy)
    balanced_accuracy_std = np.std(balanced_accuracy)
    print(f"{'Accuracy (bal.)':9s}\tMean = {balanced_accuracy_mean:.{digits}f},\tSD = {balanced_accuracy_std:.{digits}f}")
