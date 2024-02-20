# %%
import yaml

from radiomics import featureextractor
# import SimpleITK as sitk
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd
import utils

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from Dataset import HAM10000


# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# %%
df = pd.read_csv(config['dir']['csv'])
df = utils.insert_paths_df(df, config['dir']['img'], config['dir']['seg'])
# %%
dataset = HAM10000(df=df, transform=None)
# %%
# Read Images
# mask = mpimg.imread(ma_path,)
# mask = mask.astype(np.uint8)
# masked = masked = np.ma.masked_where(mask == 0, mask)

# img = mpimg.imread(im_path)
 
# Output Images
# fig, ax = plt.subplots(1,1)
# ax.imshow(img)
# ax.imshow(masked, alpha=0.25, cmap='winter')

# Instantiate extractor with parameter file
extractor = featureextractor.RadiomicsFeatureExtractor('params.yml')
# print("\nsettings")
# for key, value in extractor.settings.items():
#     print(f"{key}: {value}")
# print("\nenabled features")
# for key, value in extractor.enabledFeatures.items():
#     print(f"{key}: {value}")

# %%
# Set path to mask
label = 255  # Change this if the ROI in your mask is identified by a different value

# Load the image and extract a color channel
color_channel = 0
# im = sitk.ReadImage(im_path)
# selector = sitk.VectorIndexSelectionCastImageFilter()
# selector.SetIndex(color_channel)
# im = selector.Execute(im)


# Run the extractor
# results = extractor.execute(im, ma_path, label=label)
result = {}

# for key in results.keys():
#     result[key] = [results[key]]
#     # print(results[key])
# df = pd.DataFrame(result)
# # df.columns
# %%