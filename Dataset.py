import os
import typing
import logging

import torch
import pandas as pd
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset
from utils import pretty_print_dict

logger = logging.getLogger(__name__)


class HAM10000(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 transform=None):
        logger.info('Initialising ...')
        
        self._df = df
        self._transform = transform
        self.mapping_handler = MappingHandler()
        msg = 'Finished initialising - class distribution:'
        msg += pretty_print_dict(Counter(self._df['dx']))

        logger.info(msg)

    def __getitem__(self, index):
        
        # Load images from paths, map label
        image = Image.open(self._df.iloc[index]['img_path'])
        mask = Image.open(self._df.iloc[index]['seg_path'])
        label = self._df.iloc[index]['dx']
        label = torch.tensor(self.mapping_handler.convert(label))

        if self._transform:
            image = self._transform(image)
            mask = self._transform(mask)

        return image, mask, label
    
    def __len__(self):
        return len(self._df)


class MappingHandler:
    def __init__(self):
        self.mapping = {
            "MEL": 0,
            "NV": 1,
            "BCC": 2,
            "AKIEC": 3,
            "BKL": 4,
            "DF": 5,
            "VASC": 6
            }

    def convert(self, arg):
        if type(arg) == str:
            arg = arg.upper()
        if arg in self.mapping:
            # Argument is a key, return the corresponding value
            logger.info('Label converted [key->value]')
            return self.mapping[arg]
        elif arg in set(self.mapping.values()):
            # Argument is a value, return the corresponding key
            for k, v in self.mapping.items():
                if v == arg:
                    logger.info('Label converted [value->key]')
                    return k

        else:
            raise ValueError("Invalid key or value")