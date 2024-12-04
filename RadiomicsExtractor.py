import cv2
import numpy as np
import SimpleITK as sitk
import multiprocessing
import time
import os
from tqdm import tqdm, trange
import logging
from multiprocessing import Pool
from radiomics import featureextractor
from utils import pretty_dict_str
import albumentations as A
logger = logging.getLogger(__name__)


class RadiomicsExtractor():
    """
    A class for extracting radiomics features from medical (gray scale) images.
    Only 2D images are supported. Only single label segmentations are supported.

    Args:
        param_file (str): The path to the parameter file used by the RadiomicsFeatureExtractor.
        transforms (optional): A transformation function or pipeline to apply to the images and segmentations before feature extraction.

    Attributes:
        extractor (RadiomicsFeatureExtractor): The RadiomicsFeatureExtractor object used for feature extraction.
        transforms (optional): The transformation function or pipeline applied to the images and segmentations.

    Methods:
        extract_radiomics: Extracts radiomics features from an image and its corresponding segmentation.
        parallell_extraction: Performs parallel extraction of radiomics features from a list of images.
        serial_extraction: Performs serial extraction of radiomics features from a list of images.
    """

    def  __init__(self, param_file: str, transforms=None, remove_hair=True):
        self.extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
        msg = "\n\nEnabled Image Types:"
        msg += pretty_dict_str(self.extractor.enabledImagetypes, key_only=True)
        msg += "\n\nEnabled Features:"
        msg += pretty_dict_str(self.extractor.enabledFeatures, key_only=True)
        logger.info(msg)
        self.transforms = transforms
        self.remove_hair = remove_hair
        if self.transforms:
            logger.info(f"Transforms: {self.transforms}")
        if self.remove_hair:
            logger.info(f"Hair removal: {self.remove_hair}")    
    def get_enabled_image_types(self):
        return list(self.extractor.enabledImagetypes.keys())
    
    def get_enabled_features(self):
        return list(self.extractor.enabledFeatures.keys())

    def extract_radiomics(self, d:dict):
        gray_features, rgb_features = True, True
        label = self.extractor.settings.get('label', None)

        img_path = d['img_path']
        seg_path = d['seg_path']
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        elif not os.path.exists(seg_path):
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        
        im = sitk.ReadImage(img_path)
        im = sitk.GetArrayFromImage(im)
        
        sg = sitk.ReadImage(seg_path)
        sg = sitk.GetArrayFromImage(sg)
        


        if len(sg.shape) == 3:
            sg = sg[:,:,0]
        if len(sg.shape) == 2:
            sg = np.expand_dims(sg, axis=2)

        if im is None:
            raise ValueError(f"Image loading failed. Check the file path: {img_path}")
        elif sg is None:
            raise ValueError(f"Segmentation loading failed. Check the file path: {seg_path}")

        if self.remove_hair:
            im = self._hair_removal(im)
        else:
            pass
        if self.transforms:
            # print(im.shape, sg.shape)
            if im.shape[0:2] != sg.shape[0:2]:
                for tf in self.transforms:
                    if isinstance(tf, A.Resize):
                        im = tf(image=im)['image']
                        sg = tf(image=sg)['image']
            transformed = self.transforms(image=im, mask=sg)
            im = transformed['image']
            sg = transformed['mask']
            sg = sitk.GetImageFromArray(sg)

        # if len(sg.shape) == 3:
        #     sg = sitk.GetImageFromArray(transformed['mask'][:,:,0])
        # else:
        #     sg = sitk.GetImageFromArray(transformed['mask'])
        # else:
        #     if len(sg.shape) != 2:
        #         sg = sitk.GetImageFromArray(sg[:,:,0])
        #     else:
        #         sg = sitk.GetImageFromArray(sg)

        if gray_features:
            im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im_gray = np.expand_dims(im_gray, axis=2)
            im_gray = sitk.GetImageFromArray(im_gray)
            features = self.extractor.execute(im_gray, sg, label=label)
            features_gray = features
        if rgb_features:
            r, g, b = cv2.split(im)
            r = np.expand_dims(r, axis=2)
            g = np.expand_dims(g, axis=2)
            b = np.expand_dims(b, axis=2)
            r = sitk.GetImageFromArray(r)
            g = sitk.GetImageFromArray(g)
            b = sitk.GetImageFromArray(b)

            dicts = [self.extractor.execute(r, sg, label=label),
                     self.extractor.execute(g, sg, label=label),
                     self.extractor.execute(b, sg, label=label)]
 
            big_dict = {}
            for i, d in enumerate(dicts):
                for k, v in d.items():
                    new_key = f"{k}_r" if i == 0 else f"{k}_g" if i == 1 else f"{k}_b"
                    if new_key in big_dict:
                        big_dict[new_key].extend(v)
                    else:
                        big_dict[new_key] = v
            features = big_dict
    
            if gray_features:
                features.update(features_gray)

        return features
    

    def _img_svd(self, img, k=90):
        u, s, v = np.linalg.svd(img, full_matrices=False)
        img = np.dot(u[:, :k] * s[:k], v[:k, :]).astype(np.uint8)
        return img


    def parallell_extraction(self, list_of_dicts: list, n_processes = None):
        logger.info("Extraction mode: parallel")
        if n_processes is None:
            n_processes = multiprocessing.cpu_count() - 1
        start_time = time.time()
        with Pool(n_processes) as pool:
            results = list(tqdm(pool.imap(self.extract_radiomics, list_of_dicts),
                                 total=len(list_of_dicts)))
        end_time = time.time()

        h, m, s = self._convert_time(start_time, end_time)
        logger.info(f" Time taken: {h}h:{m}m:{s}s")

        return results
    

    def serial_extraction(self, list_of_dicts: list):
        logger.info("Extraction mode: serial")
        all_results = []
            # for item in trange(len(train_df)):
        start_time = time.time()
        for item in trange(len(list_of_dicts)):
            all_results.append(self.extract_radiomics(list_of_dicts[item]))
        end_time = time.time()

        h, m, s = self._convert_time(start_time, end_time)
        logger.info(f" Time taken: {h}h:{m}m:{s}s")
        return all_results


    def _convert_time(self, start_time, end_time):
        '''
        Converts time in seconds to hours, minutes and seconds.
        '''
        dt = end_time - start_time
        h, m, s = int(dt // 3600), int((dt % 3600 ) // 60), int(dt % 60)
        return h, m, s
    

    def _hair_removal(self, im, to_gray=False):
        '''
        Remove hair (and similar objects like ruler marks) from the image using inpainting algorithm.
        '''
        kernel = cv2.getStructuringElement(1, (17, 17)) # Kernel for the morphological filtering
        src = im
        grayScale = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) #1 Convert the original image to grayscale
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel) #2 Perform the blackHat filtering on the grayscale image to find the hair countours
        ret,thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY) # intensify the hair countours in preparation for the inpainting algorithm
        dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA) # inpaint on the original image
        if to_gray:
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        return dst