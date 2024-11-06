import os
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
# %%

path = '/media/dysk_a/jr_buler/HAM10000/test/ISIC2018_Task3_Test_Images'
save_path = '/media/dysk_a/jr_buler/HAM10000/test/ISIC2018_Task3_Test_Images_resized'

# Create the save path directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Get a list of all files in the directory
files = os.listdir(path)

# Iterate over the files and load each image
for file in files:
    # Skip if file is .DS_Store
    if file == '.DS_Store':
        continue
    
    # Construct the full file path
    file_path = os.path.join(path, file)
    
    # Load the image
    image = Image.open(file_path)           
    image = np.array(image)

    # print(image.shape)
    transform = A.Resize(256, 256)
    resized_image = transform(image=image)
    # print(resized_image['image'].shape)
    resized_image = resized_image['image']

    save_file_path = os.path.join(save_path, file)
    # Save the resized image
    resized_image = Image.fromarray(resized_image)
    resized_image.save(save_file_path)
    
# %%