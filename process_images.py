#%%
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt


dir_imag = "/media/dysk_a/jr_buler/HAM10000/test/ISIC2018_Task3_Test_Images_resized"
dir_to_save = "/media/dysk_a/jr_buler/HAM10000/test/ISIC2018_Task3_Test_Images_resized_no_hair"

def plot_image(dir_imag, index):
    files_imag = sorted(os.listdir(dir_imag))
    image = cv2.imread(os.path.join(dir_imag, files_imag[index]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original image')
    image_no_hair = hair_removal(image)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(image_no_hair)
    plt.title('No hair image')
    plt.show()

def save_image(dir_imag, dir_to_save, file):
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
        print("Directory ", dir_to_save, " Created ")
    image = cv2.imread(os.path.join(dir_imag, file), cv2.IMREAD_COLOR)
    processed_image = hair_removal(image)
    # print("Saving image ", end='~ ')
    # print(os.path.join(dir_to_save, file))
    cv2.imwrite(os.path.join(dir_to_save, file), processed_image)

def hair_removal(im, to_gray=False):
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

# %%
# plots
if 0:
    for item in range(20):
        plot_image(dir_imag, item)

# saving
if 1:
    for file in sorted(os.listdir(dir_imag)):
        save_image(dir_imag, dir_to_save, file)
# %%