# %% IMPORTS AND SETTINGS
import yaml

from radiomics import featureextractor
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import utils

import logging
from Dataset import HAM10000
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


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
df = utils.group_df(df)
train_df, val_df, test_df = utils.random_split_df(df,
                                                  config['dataset']['split_fraction_train_rest'],
                                                  config['dataset']['split_fraction_val_test'],
                                                  seed=seed)
train_df = utils.ungroup_df(train_df)
val_df = utils.ungroup_df(val_df)
test_df = utils.ungroup_df(test_df)

train_df = utils.insert_paths_df(train_df, config['dir']['img'], config['dir']['seg'])
val_df = utils.insert_paths_df(val_df, config['dir']['img'], config['dir']['seg'])
test_df = utils.insert_paths_df(test_df, config['dir']['img'], config['dir']['seg'])

# %%
transform = None
train_ds = HAM10000(df=train_df, transform=transform)
val_ds = HAM10000(df=val_df, transform=transform)
test_ds = HAM10000(df=test_df, transform=transform)
# %% RADIOMICS FEATURES EXTRACTION [ON-LINE OFF-LINE?]


# extractor = featureextractor.RadiomicsFeatureExtractor('params.yml')
# results = extractor.execute(im, ma_path, label=label)

# %%
# Set path to mask
label = 255  # Change this if the ROI in your mask is identified by a different value
color_channel = 0
# im = sitk.ReadImage(im_path)
# selector = sitk.VectorIndexSelectionCastImageFilter()
# selector.SetIndex(color_channel)
# im = selector.Execute(im)



# %%