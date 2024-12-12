# %% IMPORTS AND SETTINGS
import pandas as pd
import yaml
import utils
import pickle
import logging
import torch
import numpy as np
import albumentations as A
from Dataset import HAM10000
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.augmentations import ClassificationSMOTE

from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')

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
device = config['device'] if torch.cuda.is_available() else 'cpu'

# %%

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


img_size = config['dataset']['img_size']
transforms_train = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
    ])

transforms_val_test = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
    ])


# %% Tabnet

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
fold_results = []
k = config['dataset']['k_folds']
# kf = KFold(n_splits=k, shuffle=True, random_state=seed)
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
df = pd.concat([train_df, val_df], ignore_index=True)
kf.get_n_splits(df)


for fold_index, (train_indices, val_indices) in enumerate(kf.split(df, df['dx'])):
    train_fold = df.iloc[train_indices]
    val_fold = df.iloc[val_indices]
    
    train_ds = HAM10000(df=train_fold, transform=transforms_train, mode=config['dataset']['mode'])
    val_ds = HAM10000(df=val_fold, transform=transforms_val_test, mode=config['dataset']['mode'])


    pretrainer = TabNetPretrainer()
    pretrainer.fit()
    model = TabNetClassifier()
    model.fit()

    # model test

# %%