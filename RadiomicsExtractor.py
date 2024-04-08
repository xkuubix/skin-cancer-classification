from radiomics import featureextractor
import SimpleITK as sitk
import multiprocessing
import time
from multiprocessing import Pool
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm, trange
from utils import pretty_dict_str
import cv2


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
        img_path = d['img_path']
        seg_path = d['seg_path']
        # tutaj augmentacje
        # im = sitk.ReadImage(img_path)
        # sg = sitk.ReadImage(seg_path)
        if self.remove_hair:
            im = self._hair_removal(img_path)
        else:
            im = cv2.imread(img_path)
        sg = cv2.imread(seg_path)
        if self.transforms:
            # im = sitk.GetArrayFromImage(im)
            # sg = sitk.GetArrayFromImage(sg)
            transformed = self.transforms(image=im, mask=sg)
            # select color channel if applicable
            # (somehow sitk reads single channel mask in 3 (duplicates) channels)
            if len(im.shape) != 2:
                im = sitk.GetImageFromArray(transformed['image'][:,:,0])
            else:
                im = sitk.GetImageFromArray(transformed['image'])
            if len(sg.shape) != 2:
                sg = sitk.GetImageFromArray(transformed['mask'][:,:,0])
            else:
                sg = sitk.GetImageFromArray(transformed['mask'])
        label = self.extractor.settings.get('label', None)
        return self.extractor.execute(im, seg_path, label=label)
    
    def parallell_extraction(self, list_of_dicts: list, n_processes = None):
        logger.info(f"Extraction mode: parallel")
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
        logger.info(f"Extraction mode: serial")
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
    
    def _hair_removal(self, im_path, to_gray=False):
        '''
        Remove hair (and similar objects like ruler marks) from the image using inpainting algorithm.
        '''
        kernel = cv2.getStructuringElement(1, (17, 17)) # Kernel for the morphological filtering
        src = cv2.imread(im_path)
        grayScale = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) #1 Convert the original image to grayscale
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel) #2 Perform the blackHat filtering on the grayscale image to find the hair countours
        ret,thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY) # intensify the hair countours in preparation for the inpainting algorithm
        dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA) # inpaint on the original image
        if to_gray:
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        return dst