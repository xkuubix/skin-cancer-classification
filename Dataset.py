import logging
import torch
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset
import albumentations as A
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
        self.mapping_handler = MappingHandler(binary=False)
        msg = 'Initialized - class distribution:'
        msg += pretty_dict_str(Counter(self._df['dx']))
        logger.info(msg)
        self.mode = mode
        if self.mode not in ['images', 'radiomics', 'hybrid']:
            raise ValueError("Invalid mode")
        logger.info(f'Dataset Mode: {self.mode}')


    def __getitem__(self, index):
        
        image_path = self._df.iloc[index]['img_path']
        segmentation_path = self._df.iloc[index]['seg_path']
        label_str = self._df.iloc[index]['dx']
        label = self.mapping_handler._convert(label_str)
        data_dict_im = {}
        data_dict_rad = {}
        if self.mode in ['images', 'hybrid']:
            # Load images from paths, map label
            image = Image.open(image_path)
            mask = Image.open(segmentation_path)
           
            image = np.array(image)
            mask = np.array(mask)

            if self._transform:
                for tft in self._transform:
                    if isinstance(tft, A.Resize):
                        image = tft(image=image)['image']
                        mask = tft(image=mask)['image']
                transformed = self._transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            image = image.transpose(2, 0, 1)
            # mask = mask.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(np.array(label)).long()
            # image = image / 255.0

            data_dict_im = {
                'image': image,
                'mask': mask,
                'label': label,
                'img_path': image_path,
                'seg_path': segmentation_path,
                'label_str': label_str
                }
        if self.mode in ['radiomics', 'hybrid']:
            label = torch.from_numpy(np.array(label)).long()
            features = self._df.iloc[index].drop(self._df.columns[0:10])
            features_names = self._df.columns[10:].to_list()
            features = torch.tensor(np.array(features, dtype=np.float32), dtype=torch.float32)
            data_dict_rad = {
                'label': label,
                'img_path': image_path,
                'seg_path': segmentation_path,
                'label_str': label_str,
                'features': features,
                'features_names': features_names
                }
        return dict(data_dict_im, **data_dict_rad)
    
    def __len__(self):
        return len(self._df)


class MappingHandler:
    def __init__(self, binary=False):
        if not binary:
            self.mapping = {
                "AKIEC": 0,
                "BCC": 1,
                "BKL": 2,
                "DF": 3,
                "MEL": 4,
                "NV": 5,
                "VASC": 6
                }
        elif binary:
            self.mapping = {
                "AKIEC": 1.,
                "BCC": 1.,
                "BKL": 0.,
                "DF": 0.,
                "MEL": 1.,
                "NV": 0.,
                "VASC": 0.
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
        
def pretty_dict_str(d, key_only=False):
    #take empty string
    sorted_list = sorted(d.items())
    sorted_dict = {}
    for key, value in sorted_list:
        sorted_dict[key] = value
    pretty_dict = ''  
     
    #get items for dict
    if key_only:
        for k, _ in sorted_dict.items():
            pretty_dict += f'\n\t{k}'
    else:
        for k, v in sorted_dict.items():
            pretty_dict += f'\n\t{k}:\t{v}'
        #return result
    return pretty_dict