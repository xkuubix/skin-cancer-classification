# %% IMPORTS AND SETTINGS
import os
import yaml
import uuid
import pickle
import joblib
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
import neptune


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
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
    ])

transforms_val_test = A.Compose([
    A.Resize(img_size, img_size),
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
device = config['device'] if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"Mode: {config['dataset']['mode']}")
if config['dataset']['mode'] in ['hybrid', 'images']:
    print(f"Backbone: {config['net_train']['backbone']}")

num_classes = 1 if config['net_train']['criterion'] == 'bce' else len(train_df['dx'].unique())
if config['net_train']['criterion'] == 'bce':
    criterion = torch.nn.BCELoss()
elif config['net_train']['criterion'] == 'ce':
    criterion = torch.nn.CrossEntropyLoss()

if config['neptune']:
    run = neptune.init_run(project='ProjektMMG/skin-lesions')
    run['parameters'] = config
    run['no radiomic features'] = len(test_df.columns[10:])
else:
    run = None

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
    UUID = uuid.uuid4().hex
    print(f"Fold {fold_index} - UUID: {UUID}")
    print(f"  Train: len={len(train_indices)}")
    print(f"  Train class distribution: {df.iloc[train_indices]['dx'].value_counts(normalize=True)}")
    print(f"  Val:  len={len(val_indices)}")
    print(f"  Validation class distribution: {df.iloc[val_indices]['dx'].value_counts(normalize=True)}")
    # feature selecct + normalization (copy df cuz it is in the loop)
    train_fold, val_fold, test_fold, save_data = utils.prepare_data_for_fold(
        train_fold.copy(), val_fold.copy(), test_df.copy(), random_state=seed)
    file_name = f"features_fold_{fold_index}_{UUID}.pth"
    if run:
        run[f'fold_{fold_index} features'] = file_name
    full_path = os.path.join(config['dir']['models'], file_name)
    joblib.dump(save_data, full_path)

    print(f"Training mean (column 11): {train_fold.iloc[:, 10].mean()}")
    print(f"Validation mean (column 11): {val_fold.iloc[:, 10].mean()}")
    print(f"Test mean (column 11): {test_fold.iloc[:, 10].mean()}")

    # datasets & dataloaders
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
                                        backbone=config['net_train']['backbone'])
    elif config['dataset']['mode'] == 'radiomics':
        model = RadiomicsClassifier(radiomic_feature_size=len(train_ds[0]["features_names"]),
                                    num_classes=num_classes)
    elif config['dataset']['mode'] == 'images':
        model = ImageClassifier(num_classes=num_classes,
                                backbone=config['net_train']['backbone'])
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
    if run:
        run['fold'] = fold_index
    # Train
    net_dict = train_net(model, train_dl, val_dl,
                         criterion, optimizer,
                         config, device, run, fold_index)
    model.load_state_dict(net_dict)
    # Test
    fold_result = test_net(model, test_dl, config, device, mapping_handler, run, fold_index)
    fold_results.append(fold_result)


    file_name = f"model_fold_{fold_index}_{UUID}.pth"
    full_path = os.path.join(config['dir']['models'], file_name)
    torch.save(net_dict, full_path)
    if run:
        run[f'fold_{fold_index} model'] = file_name

utils.print_metrics(fold_results, run)

if config['neptune']:
    run.stop()