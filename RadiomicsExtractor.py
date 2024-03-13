from radiomics import featureextractor
import SimpleITK as sitk
import multiprocessing
import time
from multiprocessing import Pool
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm, trange
from utils import pretty_print_dict

class RadiomicsExtractor():
    def  __init__(self, param_file: str):
        self.extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
        msg = "\n\nEnabled Image Types:"
        msg += pretty_print_dict(self.extractor.enabledImagetypes, key_only=True)
        msg += "\n\nEnabled Features:"
        msg += pretty_print_dict(self.extractor.enabledFeatures, key_only=True)
        logger.info(msg)

    
    def extract_radiomics(self, d:dict, label=255, color_channel=0):
        img_path = d['img_path']
        seg_path = d['seg_path']
        # tutaj augmentacje
        im = sitk.ReadImage(img_path)
        if im.GetNumberOfComponentsPerPixel() > 1:
            selector = sitk.VectorIndexSelectionCastImageFilter()
            selector.SetIndex(color_channel)
            im = selector.Execute(im)
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