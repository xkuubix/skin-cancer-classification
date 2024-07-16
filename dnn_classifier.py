# %% IMPORTS AND SETTINGS
import yaml
import torch
import numpy as np
import pandas as pd
import utils
import logging
import torch.nn as nn
import albumentations as A
from Dataset import HAM10000
from Dataset import MappingHandler
from torchvision import models
from torch.utils.data import WeightedRandomSampler
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

# TODO dodać neptune albo w&b
from utils import train_net, test_net


logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
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
# Transforms
img_size = config['dataset']['img_size']
crop_size = 400
transforms_train = A.Compose([
    # A.Affine(scale=(0.9, 1),
    #          shear=None,
    #          rotate=(-45, 45),
    #          p=1.),
    A.CenterCrop(crop_size, crop_size, p=1.),
    A.Resize(img_size, img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.ShiftScaleRotate(p=0.5),
    # A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), p=0.5),
    # A.RandomGamma(gamma_limit=(95, 105), p=0.5),
    # A.GridDistortion(distort_limit=0.1, p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.OpticalDistortion(p=0.5),
    # A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
    
    # A.ElasticTransform(p=0.5),
    # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), p=1.0),
])
transforms_val_test = A.Compose([A.Resize(img_size, img_size),
                                #  A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), p=1.0)
                                ])
                                 

# Data split
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

# train_d = train_df.to_dict(orient='records')
# train_d = utils.oversample_data(train_d)
# train_df = pd.DataFrame(train_d)
# Dataset
train_ds = HAM10000(df=train_df, transform=transforms_train, mode='images')
val_ds = HAM10000(df=val_df, transform=transforms_val_test ,mode='images')
test_ds = HAM10000(df=test_df, transform=transforms_val_test, mode='images')

# Sampler
mapping_handler = MappingHandler().mapping
target = [mapping_handler[label.upper()] for label in train_df['dx'].values]
class_sample_count = np.array(
    [len(np.where(target == t)[0]) for t in np.unique(target)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in target])

samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight,
                                len(samples_weight), # num_samples drawn each epoch
                                replacement=True)
# Data loaders
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=config['net_train']['batch_size'],
    sampler=sampler,
    # shuffle=True,
    pin_memory=True, num_workers=14)
val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size=config['net_train']['batch_size'],
    shuffle=False, pin_memory=True, num_workers=14)
test_dl = torch.utils.data.DataLoader(
    test_ds, batch_size=config['net_train']['batch_size'],
    shuffle=False, pin_memory=True, num_workers=14)

# %%
# TRAINING LOOP --------------------------------
# Set device
if torch.cuda.is_available():
    device = config['device']
else:
    device = 'cpu'
print(f"Device: {device}")

# Set network, criterion, optimizer, scheduler
# net = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_classes = len(train_df['dx'].unique())
# num_features = net.fc.in_features
# net.fc = nn.Linear(num_features, num_classes)

# Create capsule network.
from model import FixCapsNet
n_channels = 3
conv_outputs = 128 #Feature_map
num_primary_units = 8
primary_unit_size = 16 * 6 * 6  # fixme get from conv2d
output_unit_size = 16
mode='DS'
net = FixCapsNet(conv_inputs=n_channels,
                 conv_outputs=conv_outputs,
                 primary_units=num_primary_units,
                 primary_unit_size=primary_unit_size,
                 num_classes=num_classes,
                 output_unit_size=16,
                 init_weights=True,
                 mode=mode)
net.to(device=device)
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(),
                             lr=config['net_train']['lr'],
                             weight_decay=config['net_train']['wd'])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# %%
# print(f"Optimizer: {optimizer}")
torch.autograd.set_detect_anomaly(True)
net_dict = train_net(net, train_dl, val_dl, criterion, optimizer, num_classes, config, device)
net.load_state_dict(net_dict)
test_net(net, test_dl, num_classes, device)
# %%