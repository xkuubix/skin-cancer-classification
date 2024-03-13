from radiomics import featureextractor
import SimpleITK as sitk
import multiprocessing
import time
from multiprocessing import Pool
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm, trange

class RadiomicsExtractor():
    def  __init__(self, param_file: str):
        self.extractor = featureextractor.RadiomicsFeatureExtractor(param_file)

    
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
        if n_processes is None:
            n_processes = multiprocessing.cpu_count() - 1
        start_time = time.time()
        with Pool(n_processes) as pool:
            # results = pool.map(self.extract_radiomics, list_of_dicts, chunksize=None)
            results = list(tqdm(pool.imap(self.extract_radiomics, list_of_dicts,
                                          chunksize=n_processes),
                                 total=len(list_of_dicts)))


        end_time = time.time()
        logger.info(f" {(end_time - start_time) / len(list_of_dicts):.3f} s/img")
        logger.info(f" {end_time - start_time:.3f} s\n")
        return results
    
    def serial_extraction(self, list_of_dicts: list):
        all_results = []
            # for item in trange(len(train_df)):
        for item in trange(len(list_of_dicts)):
            all_results.append(self.extract_radiomics(list_of_dicts[item]))