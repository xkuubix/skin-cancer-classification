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
transforms_train = A.Compose([
    # A.Affine(scale=(0.9, 1),
    #          shear=None,
    #          rotate=(-45, 45),
    #          p=1.),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), p=0.5),
    # A.RandomGamma(p=0.5 ),
    # A.GridDistortion(p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.ElasticTransform(p=0.5),
    # A.OpticalDistortion(p=0.5),
    # A.RandomBrightnessContrast(p=0.5),
    A.Resize(224, 224),
])
transforms_val_test = A.Resize(224, 224)

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
    shuffle=True,
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
net = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_classes = len(train_df['dx'].unique())
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, num_classes)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(),
                             lr=config['net_train']['lr'],
                             weight_decay=config['net_train']['wd'])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# %%
print(f"Optimizer: {optimizer}")
net_dict = train_net(net, train_dl, val_dl, criterion, optimizer, config, device)
net.load_state_dict(net_dict)
test_net(net, test_dl, device)