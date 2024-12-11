import os
import numpy as np
from PIL import Image
import albumentations as A
# %%

'''
img TRAIN ->                 450 x 600 -> 450 x 450
seg TRAIN ->                 450 x 600 -> 450 x 450

img TEST ->                      450 x 600 -> 450 x 450
seg TEST ->         224 x 224 -> 450 x 600 -> 450 x 450

2x for HAIR and NO_HAIR

'''

img_path = '/media/dysk_a/jr_buler/HAM10000/test/ISIC2018_Task3_Test_Images'
seg_path = '/media/dysk_a/jr_buler/HAM10000/test/ISIC2018_Task3_Test_Images_Segmentations_trained_on_HAM'

img_save_path = '/media/dysk_a/jr_buler/HAM10000/test/new/ISIC2018_Task3_Test_Images'
seg_save_path = '/media/dysk_a/jr_buler/HAM10000/test/new/ISIC2018_Task3_Test_Images_Segmentations_trained_on_HAM'

os.makedirs(img_save_path, exist_ok=True)
os.makedirs(seg_save_path, exist_ok=True)


image_transform = A.CenterCrop(450, 450)
mask_upscale_and_crop = A.Compose([
    A.Resize(450, 600),
    A.CenterCrop(450, 450)
])

def process_images(input_path, output_path, transform):
    files = os.listdir(input_path)
    
    for file in files:
        if file == '.DS_Store':
            continue
        
        file_path = os.path.join(input_path, file)

        image = Image.open(file_path)           
        image = np.array(image)

        transformed = transform(image=image)
        processed_image = transformed['image']

        save_file_path = os.path.join(output_path, file)
        processed_image = Image.fromarray(processed_image)
        processed_image.save(save_file_path)


def process_masks(input_path, output_path, transform):
    files = os.listdir(input_path)
    
    for file in files:
        if file == '.DS_Store':
            continue
        
        file_path = os.path.join(input_path, file)
        
        mask = Image.open(file_path)           
        mask = np.array(mask)
        
        transformed = transform(image=mask)
        processed_mask = transformed['image']

        save_file_path = os.path.join(output_path, file)
        processed_mask = Image.fromarray(processed_mask)
        processed_mask.save(save_file_path)


process_images(img_path, img_save_path, image_transform)
process_masks(seg_path, seg_save_path, mask_upscale_and_crop)
    
# %%
