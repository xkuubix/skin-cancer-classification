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
    # A.Normalize([0.76530149, 0.54760609, 0.5719637], [0.14010777, 0.15290574, 0.17048959],
                # always_apply=True),
    # A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True)
    ])

transforms_val_test = A.Compose([
    A.Resize(img_size, img_size),
    # A.Normalize([0.76530149, 0.54760609, 0.5719637], [0.14010777, 0.15290574, 0.17048959],
                # always_apply=True),
    # A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
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
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

num_classes = 1 if config['net_train']['criterion'] == 'bce' else len(train_df['dx'].unique())
if config['net_train']['criterion'] == 'bce':
    criterion = torch.nn.BCELoss()
elif config['net_train']['criterion'] == 'ce':
    criterion = torch.nn.CrossEntropyLoss()
# %% ---------------------K-Fold Cross Validation---------------------
fold_results = []
k = config['dataset']['k_folds']
# kf = KFold(n_splits=k, shuffle=True, random_state=seed)
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
# Concatenate train and validation set for k-fold cross validation
df = pd.concat([train_df, val_df], ignore_index=True)
kf.get_n_splits(df)
for fold_index, (train_indices, val_indices) in enumerate(kf.split(df, df['dx'])):
    train_fold = df.iloc[train_indices]
    val_fold = df.iloc[val_indices]
    print('--------------------------------')
    print(f"Fold {fold_index}:")
    print(f"  Train: len={len(train_indices)}")
    print(f"  Train class distribution: {df.iloc[train_indices]['dx'].value_counts(normalize=True)}")
    print(f"  Val:  len={len(val_indices)}")
    print(f"  Validation class distribution: {df.iloc[val_indices]['dx'].value_counts(normalize=True)}")
    # Prepare data for the fold (copy df cuz it is in the loop)
    train_fold, val_fold, test_fold = utils.prepare_data_for_fold(
        train_fold, val_fold, test_df.copy(), random_state=seed)

    # Create datasets and dataloaders
    train_ds = HAM10000(df=train_fold, transform=transforms_train, mode=config['dataset']['mode'])
    val_ds = HAM10000(df=val_fold, transform=transforms_val_test, mode=config['dataset']['mode'])
    test_ds = HAM10000(df=test_fold, transform=transforms_val_test, mode=config['dataset']['mode'])
    
    mapping_handler = MappingHandler().mapping
    labels = [int(mapping_handler[label.upper()]) for label in train_fold['dx'].values]
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=config['net_train']['batch_size'],
        sampler=utils.generate_sampler(labels),
        pin_memory=True, num_workers=config['net_train']['num_workers'])
    
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=config['net_train']['batch_size'],
        shuffle=False, pin_memory=True, num_workers=config['net_train']['num_workers'])

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=config['net_train']['batch_size'],
        shuffle=False, pin_memory=True, num_workers=config['net_train']['num_workers'])
    
    # new instance of the model for each fold
    if config['dataset']['mode'] == 'hybrid':
        model = DeepRadiomicsClassifier(radiomic_feature_size=len(train_ds[0]["features_names"]),
                                        num_classes=num_classes,
                                        backbone='resnet34')
    elif config['dataset']['mode'] == 'radiomics':
        model = RadiomicsClassifier(radiomic_feature_size=len(train_ds[0]["features_names"]),
                                    num_classes=num_classes)
    elif config['dataset']['mode'] == 'images':
        model = ImageClassifier(num_classes=num_classes,
                                backbone='resnet34')
    model.to(device=device)

    # new optimizer for each fold
    if config['net_train']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config['net_train']['lr'],
                                     weight_decay=config['net_train']['wd'])
    elif config['net_train']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config['net_train']['lr'],
                                    weight_decay=config['net_train']['wd'])

    # Train
    net_dict = train_net(model, train_dl, val_dl, criterion, optimizer, num_classes, config, device)
    model.load_state_dict(net_dict)
    # Test
    fold_result = test_net(model, test_dl, config, device)
    fold_results.append(fold_result)

utils.print_metrics(fold_results)
