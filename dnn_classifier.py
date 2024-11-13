# %% IMPORTS AND SETTINGS
import yaml
import pickle
import torch
import numpy as np
import pandas as pd
import utils
from net_utils import train_net, test_net
import logging
import albumentations as A
from Dataset import HAM10000, MappingHandler
from sklearn.model_selection import StratifiedKFold  #, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from model import DeepRadiomicsClassifier, RadiomicsClassifier, ImageClassifier
# TODO dodać neptune albo w&b


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# DATASET --------------------------------
img_size = config['dataset']['img_size']
crop_size = 256
transforms_train = A.Compose([
    A.RandomResizedCrop(crop_size, crop_size, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.GridDistortion(distort_limit=0.1, p=0.5),
    A.ColorJitter(p=0.5),
    A.GaussNoise(p=0.5),
    A.Resize(img_size, img_size),
    A.Normalize([0.76530149, 0.54760609, 0.5719637], [0.14010777, 0.15290574, 0.17048959],
                always_apply=True),
    # A.Resize(img_size, img_size),
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True)
    ])

transforms_val_test = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize([0.76530149, 0.54760609, 0.5719637], [0.14010777, 0.15290574, 0.17048959],
                always_apply=True),
    # A.Resize(img_size, img_size),
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True)
    ])

# Dataset
with open(config['dir']['pkl_train'], 'rb') as handle:
    train_df = pickle.load(handle)
    logger.info(f"Loaded radiomics features (train) from {config['dir']['pkl_train']}")
with open(config['dir']['pkl_val'], 'rb') as handle:
    val_df = pickle.load(handle)
    logger.info(f"Loaded radiomics features (val) from {config['dir']['pkl_val']}")
with open(config['dir']['pkl_test'], 'rb') as handle:
    test_df = pickle.load(handle)
    logger.info(f"Loaded radiomics features (test) from {config['dir']['pkl_test']}")

mapping_handler = MappingHandler().mapping
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

num_classes = 1 if config['net_train']['criterion'] == 'bce' else len(train_df['dx'].unique())

df = pd.concat([train_df, val_df], ignore_index=True)
scaler = StandardScaler()

# %%
lasso = LassoCV(cv=5, random_state=seed,
                max_iter=10000, tol=0.001)
X = scaler.fit_transform(df[df.columns[10:]])
y = [int(mapping_handler[label.upper()]) for label in df['dx'].values]
lasso.fit(X, y)

selected_features = df.columns[10:][lasso.coef_ != 0]

radiomic_featre_size = len(selected_features)
print(f"Selected {radiomic_featre_size} features")
    
if config['net_train']['criterion'] == 'bce':
    criterion = torch.nn.BCELoss()
elif config['net_train']['criterion'] == 'ce':
    criterion = torch.nn.CrossEntropyLoss()

# %%
# Select only features selected by Lasso and Standardize
df[df.columns[10:]] = scaler.transform(df[df.columns[10:]])
first_10_columns = df.columns[:10]
remaining_columns = [col for col in df.columns[10:] if col in selected_features]
columns_to_keep = list(first_10_columns) + remaining_columns
df = df[columns_to_keep]

test_df[test_df.columns[10:]] = scaler.transform(test_df[test_df.columns[10:]])
df2 = test_df
first_10_columns = df2.columns[:10]
remaining_columns = [col for col in df2.columns[10:] if col in selected_features]
columns_to_keep = list(first_10_columns) + remaining_columns
test_df = df2[columns_to_keep]
# %%
# K-Fold Cross Validation
fold_results = []
k = config['dataset']['k_folds']
# kf = KFold(n_splits=k, shuffle=True, random_state=seed)
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
kf.get_n_splits(df)
for i, (train_idx, val_idx) in enumerate(kf.split(df, df['dx'])):
    train_fold = df.iloc[train_idx]
    val_fold = df.iloc[val_idx]
    print('--------------------------------')
    print(f"Fold {i}:")
    print(f"  Train: len={len(train_idx)}")
    print(df.iloc[train_idx]['dx'].value_counts(normalize=True))        
    print(f"  Val:  len={len(val_idx)}")
    print(df.iloc[val_idx]['dx'].value_counts(normalize=True))
    train_ds = HAM10000(df=train_fold, transform=transforms_train, mode=config['dataset']['mode'])
    val_ds = HAM10000(df=val_fold, transform=transforms_val_test, mode=config['dataset']['mode'])
    target = [int(mapping_handler[label.upper()]) for label in train_fold['dx'].values]
    sampler = utils.generate_sampler(target)
    # Create train and validation data loaders
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config['net_train']['batch_size'],
        sampler=sampler,
        pin_memory=True, num_workers=5)
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config['net_train']['batch_size'],
        shuffle=False, pin_memory=True, num_workers=5)
    test_ds = HAM10000(df=test_df, transform=transforms_val_test, mode=config['dataset']['mode'])
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=config['net_train']['batch_size'],
        shuffle=False, pin_memory=True, num_workers=5)

    # Create a new instance of the model for each fold
    if config['dataset']['mode'] == 'hybrid':
        model = DeepRadiomicsClassifier(radiomic_feature_size=len(train_ds[0]["features_names"]),
                                        num_classes=num_classes,
                                        backbone='efficientnet_b0')
    elif config['dataset']['mode'] == 'radiomics':
        model = RadiomicsClassifier(radiomic_feature_size=len(train_ds[0]["features_names"]),
                                    num_classes=num_classes)
    elif config['dataset']['mode'] == 'images':
        model = ImageClassifier(num_classes=num_classes,
                                backbone='efficientnet_b0')
    
    model.to(device=device)

    # Create a new optimizer for each fold
    if config['net_train']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config['net_train']['lr'],
                                     weight_decay=config['net_train']['wd'])
    elif config['net_train']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config['net_train']['lr'],
                                    weight_decay=config['net_train']['wd'])

    # Train the model
    net_dict = train_net(model, train_dl, val_dl, criterion, optimizer, num_classes, config, device)
    model.load_state_dict(net_dict)

    # Test the model on the test set
    fold_result = test_net(model, test_dl, config, device)
    fold_results.append(fold_result)

# %%
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
        if label == 'weighted avg':
            metrics_summary_macro['precision'].append(metrics['precision'])
            metrics_summary_macro['recall'].append(metrics['recall'])
            metrics_summary_macro['f1-score'].append(metrics['f1-score'])
        elif label == 'macro avg':
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
print("\nSummary of performance metrics (weighted avg):")
for metric, values in metrics_summary_weighted.items():
    mean = np.mean(values)
    std = np.std(values)
    print(f"{metric.capitalize():9s}\tMean = {mean:.4f},\tSD = {std:.4f}")

print("\nSummary of performance metrics (macro avg):")
for metric, values in metrics_summary_macro.items():
    mean = np.mean(values)
    std = np.std(values)
    print(f"{metric.capitalize():9s}\tMean = {mean:.4f},\tSD = {std:.4f}")

accuracy_mean = np.mean(accuracy)
accuracy_std = np.std(accuracy)
print(f"\n{'Accuracy':9s}\tMean = {accuracy_mean:.4f},\tSD = {accuracy_std:.4f}")

balanced_accuracy_mean = np.mean(balanced_accuracy)
balanced_accuracy_std = np.std(balanced_accuracy)
print(f"{'Accuracy (bal.)':9s}\tMean = {balanced_accuracy_mean:.4f},\tSD = {balanced_accuracy_std:.4f}")
# %%
