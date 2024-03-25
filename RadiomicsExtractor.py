from radiomics import featureextractor
import SimpleITK as sitk
import multiprocessing
import time
from multiprocessing import Pool
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm, trange
from utils import pretty_dict_str

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

    def  __init__(self, param_file: str, transforms=None):
        self.extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
        msg = "\n\nEnabled Image Types:"
        msg += pretty_dict_str(self.extractor.enabledImagetypes, key_only=True)
        msg += "\n\nEnabled Features:"
        msg += pretty_dict_str(self.extractor.enabledFeatures, key_only=True)
        logger.info(msg)
        self.transforms = transforms
    
    def get_enabled_image_types(self):
        return list(self.extractor.enabledImagetypes.keys())
    
    def get_enabled_features(self):
        return list(self.extractor.enabledFeatures.keys())

    def extract_radiomics(self, d:dict):
        img_path = d['img_path']
        seg_path = d['seg_path']
        # tutaj augmentacje
        im = sitk.ReadImage(img_path)
        sg = sitk.ReadImage(seg_path)
        if self.transforms:
            im = sitk.GetArrayFromImage(im)
            sg = sitk.GetArrayFromImage(sg)
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
        logger.info(f"Extracting radiomics features: mode-parallel")
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
        logger.info(f"Extracting radiomics features: mode-serial")
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
        dt = end_time - start_time
        h, m, s = int(dt // 3600), int((dt % 3600 ) // 60), int(dt % 60)
        return h, m, s