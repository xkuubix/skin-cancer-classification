# %% IMPORTS AND SETTINGS
import yaml
import pickle
import os
import torch
import numpy as np
import pandas as pd
import utils
import logging
from RadiomicsExtractor import RadiomicsExtractor
from Dataset import HAM10000
import datetime
import albumentations as A
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger_radiomics = logging.getLogger("radiomics")
logger_radiomics.setLevel(logging.ERROR)

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
# %% DATA LOADING - 3-SET-SPLIT
df = pd.read_csv(config['dir']['csv'])
df = utils.insert_paths_df(df, config['dir']['img'], config['dir']['seg'])
df = utils.group_df(df)

train_df, val_df, test_df = utils.random_split_df(df,
                                                  config['dataset']['split_fraction_train_rest'],
                                                  config['dataset']['split_fraction_val_test'],
                                                  seed=seed)
# %%
train_df = utils.ungroup_df(train_df)
val_df = utils.ungroup_df(val_df)
test_df = utils.ungroup_df(test_df)
# %% RADIOMICS FEATURES EXTRACTION
train_d = train_df.to_dict(orient='records')
test_d = test_df.to_dict(orient='records')
val_d = val_df.to_dict(orient='records')
# %%

if config['dataset']['train_sampling']['method'] == 'oversample':
    train_original = train_d
    train_d = utils.oversample_data(train_d)
    transforms_train = A.Compose([
        A.Affine(scale=(0.9, 1),
                 shear=(-10, 10),
                 rotate=(-45, 45),
                 p=1.),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.GaussNoise(p=0.5),
        A.CLAHE(p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=0.1,
        #                            contrast_limit=0.1,
        #                            p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(distort_limit=0.1, p=0.7),
        ], p=0.5),
        A.Normalize(),
    ])
elif config['dataset']['train_sampling']['method'] == 'undersample':
    train_d = utils.undersample_data(train_d, seed=seed, multiplier=config['dataset']['train_sampling']['multiplier'])
    transforms_train = None
elif config['dataset']['train_sampling']['method'] == 'none':
    transforms_train = None
transforms_val_test = None
# %%
if config['radiomics']['extract']:
    extractor_train = RadiomicsExtractor(param_file='params.yml',
                                         transforms=transforms_train,
                                         remove_hair=True)
    extractor_val_test = RadiomicsExtractor(param_file='params.yml',
                                            transforms=transforms_val_test,
                                            remove_hair=True)
    if config['radiomics']['mode'] == 'parallel':
        results_train = extractor_train.parallell_extraction(train_d)
        results_val = extractor_val_test.parallell_extraction(val_d)
        results_test = extractor_val_test.parallell_extraction(test_d)
    elif config['radiomics']['mode'] == 'serial':
        results_train = extractor_train.serial_extraction(train_d)
        results_val = extractor_val_test.serial_extraction(val_d)
        results_test = extractor_val_test.serial_extraction(test_d)
    
    train_df = pd.DataFrame(train_d)
    train_df = pd.concat([train_df, pd.DataFrame(results_train)], axis=1)
    val_df = pd.DataFrame(val_d)
    val_df = pd.concat([val_df, pd.DataFrame(results_val)], axis=1)
    test_df = pd.DataFrame(test_d)
    test_df = pd.concat([test_df, pd.DataFrame(results_test)], axis=1)
    
    if config['radiomics']['save']:
        # if os.path.exists(config['dir']['pkl_train']):
        #     os.remove(config['dir']['pkl_train'])
        with open(config['dir']['pkl_train'], 'wb') as handle:
            pickle.dump(train_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #     for i in range(len(train_df)):
        #         pickle.dump(train_df.iloc[i:i+1], handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config['dir']['pkl_val'], 'wb') as handle:
            pickle.dump(val_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config['dir']['pkl_test'], 'wb') as handle:
            pickle.dump(test_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved radiomics features in {config['dir']['pkl_train']}")
        logger.info(f"Saved radiomics features in {config['dir']['pkl_val']}")
        logger.info(f"Saved radiomics features in {config['dir']['pkl_test']}")
        image_types = extractor_train.get_enabled_image_types()
        feature_types = extractor_train.get_enabled_features()
        with open(config['dir']['inf'], 'w') as file:
            current_datetime = datetime.datetime.now()
            file.write(f"Modified: {current_datetime}\n")
            file.write(yaml.dump(config))
            file.write('\n\nEnabled Image Types:\n')
            file.write('\n'.join(image_types))
            file.write('\n\nEnabled Features:\n')
            file.write('\n'.join(feature_types))
            file.write('\n\nTransforms:\n' + str(transforms_train))
            file.write('\n\nHair removal: ' + str(extractor_train.remove_hair))
        logger.info(f"Saved extraction details in {config['dir']['inf']}")
# %% Logging in constructor
train_ds = HAM10000(df=train_df, mode='radiomics')
val_ds = HAM10000(df=val_df, mode='radiomics')
test_ds = HAM10000(df=test_df, mode='radiomics')
# %%