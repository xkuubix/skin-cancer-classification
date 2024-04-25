import os
import time
import random
import typing
import logging
import argparse
import torch
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix


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



# DNN TRAINING PART --------------------------------

def train_net(net, train_dl, val_dl, criterion, optimizer, config, device):
    net.to(device)
    best_val_loss = float('inf')
    best_model = None
    patience = config['net_train']['patience']
    early_stop_counter = 0
    start_time = time.time()
    for epoch in range(config['net_train']['epochs']):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_dl):
            inputs, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(f"Batch {i+1}/{len(train_dl)}, loss: {loss.item():.3f}")
        # Validation
        net.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in val_dl:
                images, labels = data['image'].to(device), data['label'].to(device)
                outputs = net(images)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_dl)
        avg_val_loss = running_val_loss / len(val_dl)
        train_accuracy = correct / total
        val_accuracy = val_correct / val_total

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Epoch {epoch+1}, train-loss: {avg_train_loss:.3f}, val-loss: {avg_val_loss:.3f}, train-accuracy: {train_accuracy:.3f}, val-accuracy: {val_accuracy:.3f}", end=' ')
        print(f"Time taken: {total_time//60:4.0f}:{total_time%60:2.0f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = net.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                break
    
    print('Finished Training')
    return best_model

def test_net(net, test_dl, device):
    net.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for data in test_dl:
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")
    print('Classification Report:')
    print(classification_report(true_labels, predicted_labels))
    print('Confusion Matrix:')
    print(confusion_matrix(true_labels, predicted_labels))
    
    print('Finished Testing')