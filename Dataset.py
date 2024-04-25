import os
import typing
import logging
# import cv2
import torch
import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset
from utils import pretty_dict_str

logger = logging.getLogger(__name__)


class HAM10000(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 mode: str,
                 transform=None):
        """
        Initializes the Dataset object.

        Args:
            df (pd.DataFrame): The input DataFrame containing the dataset.
            mode (str): The mode of the dataset. Can be either 'images' or 'radiomics'.
            transform (optional): The transformation to be applied to the dataset in 'images' mode.

        Raises:
            ValueError: If an invalid mode is provided.

        """
        self._df = df
        self._transform = transform
        self.mapping_handler = MappingHandler()
        msg = 'Initialized - class distribution:'
        msg += pretty_dict_str(Counter(self._df['dx']))
        logger.info(msg)
        self.mode = mode
        if self.mode not in ['images', 'radiomics']:
            raise ValueError("Invalid mode")
        logger.info(f'Dataset Mode: {self.mode}')


    def __getitem__(self, index):
        
        image_path = self._df.iloc[index]['img_path']
        segmentation_path = self._df.iloc[index]['seg_path']
        label_str = self._df.iloc[index]['dx']
        label = self.mapping_handler._convert(label_str)
        if self.mode == 'images':
            # Load images from paths, map label

            # image = Image.open(image_path)
            # mask = Image.open(segmentation_path)

            # image = cv2.imread(image_path)
            # mask = cv2.imread(segmentation_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.resize(image, (224, 224))
            # mask = cv2.resize(mask, (224, 224))

            image = Image.open(image_path)
            mask = Image.open(segmentation_path)
           
            image = asarray(image)
            mask = asarray(mask)
            # print(image.shape, mask.shape)

            if self._transform:
                transformed = self._transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            image = image.transpose(2, 0, 1)
            # mask = mask.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(np.array(label)).long()
            image = image / 255.0

            data_dict = {
                'image': image,
                'mask': mask,
                'label': label,
                'img_path': image_path,
                'seg_path': segmentation_path,
                'label_str': label_str
                }
        elif self.mode == 'radiomics':

            features = self._df.iloc[index].drop(self._df.columns[0:10])
            features_names = self._df.columns[10:]
            features = torch.tensor(np.array(features, dtype=np.float32), dtype=torch.float32)
            data_dict = {
                'label': label,
                'img_path': image_path,
                'seg_path': segmentation_path,
                'label_str': label_str,
                'features': features,
                'features_names': features_names
                }
        return data_dict
    
    def __len__(self):
        return len(self._df)


class MappingHandler:
    def __init__(self):
        self.mapping = {
            "AKIEC": 0,
            "BCC": 1,
            "BKL": 2,
            "DF": 3,
            "MEL": 4,
            "NV": 5,
            "VASC": 6
            }

    def _convert(self, arg):
        if type(arg) == str:
            arg = arg.upper()
        if arg in self.mapping:
            # Argument is a key, return the corresponding value
            logger.info(f'Label converted [key->value] [{arg}->{self.mapping[arg]}]')
            return self.mapping[arg]
        elif arg in set(self.mapping.values()):
            # Argument is a value, return the corresponding key
            for k, v in self.mapping.items():
                if v == arg:
                    logger.info(f'Label converted [value->key] [{k}->{v}]')
                    return k

        else:
            raise ValueError("Invalid key or value")