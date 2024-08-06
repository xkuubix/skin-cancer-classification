# %%
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
dir_mask = "/media/dysk_a/jr_buler/HAM10000/test/ISIC2018_Task3_Test_Images_Segmentations"
dir_imag = "/media/dysk_a/jr_buler/HAM10000/test/ISIC2018_Task3_Test_Images_resized"
dir_to_save = "/media/dysk_a/jr_buler/HAM10000/test/ISIC2018_Task3_Test_Images_Segmentations_processed"

def plot_masks(path_mask, path_imag, index):
    files_mask = sorted(os.listdir(path_mask))
    files_imag = sorted(os.listdir(path_imag))
    mask = cv2.imread(os.path.join(path_mask, files_mask[index]), cv2.IMREAD_GRAYSCALE)
    mask_pre = mask
    image = cv2.imread(os.path.join(path_imag, files_imag[index]), cv2.IMREAD_GRAYSCALE)
    # Plot original mask and image
    plt.subplot(1, 3, 1)
    plt.imshow(mask_pre, cmap='gray')
    plt.axis('off')
    plt.title('Original Mask')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    # smoothening mask
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask_image_cleaned = np.zeros_like(mask)
    mask = cv2.drawContours(mask_image_cleaned, [largest_contour], -1, 255, thickness=cv2.FILLED)

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(mask, cmap='gray')
    plt.title('Processed Mask')
    plt.show()

def save_mask(path_mask, dir_to_save, file):
    
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
        print("Directory ", dir_to_save, " Created ")
    mask = cv2.imread(os.path.join(path_mask, file), cv2.IMREAD_GRAYSCALE)
    # smoothening mask
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask_image_cleaned = np.zeros_like(mask)
    mask = cv2.drawContours(mask_image_cleaned, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # print("Saving mask ", end='~ ')
    # print(os.path.join(dir_to_save, file))
    cv2.imwrite(os.path.join(dir_to_save, file), mask)

# plots
if 0:
    for item in range(20):
        plot_masks(dir_mask, dir_imag, item)

# saving
if 1:
    for file in sorted(os.listdir(dir_mask)):
        save_mask(dir_mask, dir_to_save, file)
# %%
