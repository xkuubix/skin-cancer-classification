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
train_df = utils.ungroup_df(train_df)
val_df = utils.ungroup_df(val_df)
test_df = utils.ungroup_df(test_df)
# %%
transform = None
train_ds = HAM10000(df=train_df, transform=transform)
val_ds = HAM10000(df=val_df, transform=transform)
test_ds = HAM10000(df=test_df, transform=transform)
# %% RADIOMICS FEATURES EXTRACTION
d = train_df.to_dict(orient='records')

if config['dataset']['train_sampling']['method'] == 'oversample':
    d = utils.oversample_data(d)
elif config['dataset']['train_sampling']['method'] == 'undersample':
    d = utils.undersample_data(d, seed=seed, multiplier=config['dataset']['train_sampling']['multiplier'])
elif config['dataset']['train_sampling']['method'] == 'none':
    pass

if config['radiomics']['extract']:


    transforms = A.Compose([
        A.ToGray(p=1),
        # A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RandomBrightnessContrast(p=0.5),

        A.OneOf([
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.25),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
        ], p=0.25),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.5),
    ])


    extractor = RadiomicsExtractor(param_file='params.yml', transforms=transforms )
    if config['radiomics']['mode'] == 'parallel':
        results = extractor.parallell_extraction(d)
    elif config['radiomics']['mode'] == 'serial':
        results = extractor.serial_extraction(d)
    d = pd.DataFrame(d)
    d = pd.concat([d, pd.DataFrame(results)], axis=1)
    if config['radiomics']['save_or_load'] == 'save':
        with open(config['dir']['pkl'], 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved radiomics features in {config['dir']['pkl']}")
            logger.info(f"Saved extraction details in {config['dir']['inf']}")
            image_types = extractor.get_enabled_image_types()
            feature_types = extractor.get_enabled_features()
            with open(config['dir']['inf'], 'w') as file:
                current_datetime = datetime.datetime.now()
                file.write(f"Modified: {current_datetime}\n")
                file.write(yaml.dump(config))
                file.write('\n\nEnabled Image Types:\n')
                file.write('\n'.join(image_types))
                file.write('\n\nEnabled Features:\n')
                file.write('\n'.join(feature_types))
                file.write('\n\nTransforms' + str(transforms))
elif config['radiomics']['save_or_load'] == 'load':
    assert os.path.exists(config['dir']['pkl'])
    with open(config['dir']['pkl'], 'rb') as handle:
        results = pickle.load(handle)
        logger.info(f"Loaded radiomics features from {config['dir']['pkl']}")
# %%
