# %% IMPORTS AND SETTINGS
import yaml

from radiomics import featureextractor
import SimpleITK as sitk
# import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import utils

import logging
from Dataset import HAM10000

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
# %% RADIOMICS FEATURES EXTRACTION [ON-LINE OFF-LINE?]
extractor = featureextractor.RadiomicsFeatureExtractor('params.yml')
label = 255
im = sitk.ReadImage(train_ds[0]['img_path'])
color_channel = 0
selector = sitk.VectorIndexSelectionCastImageFilter()
selector.SetIndex(color_channel)
im = selector.Execute(im)
results = extractor.execute(im, train_ds[0]['seg_path'], label=label)

# %%