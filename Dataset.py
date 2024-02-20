import os
import typing
import logging

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HAM10000(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 transform=None):
        logger.debug('Initialising dataset')
        
        self.df = df
        self.transform = transform

        logger.debug('Finished initialising dataset')

    def __getitem__(self, index):
        
        image = Image.open(self.df['image_path'][index])
        mask = Image.open(self.df['mask_path'][index])
        label = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, label
    
    def __len__(self):
        return len(self.df)
