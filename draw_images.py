import os
from PIL import Image
import matplotlib.pyplot as plt


img_path = 'path HERE'
seg_path = 'path HERE'


def show_images_and_masks(img_path, seg_path, num_samples=5):
    img_files = sorted(os.listdir(img_path))
    seg_files = sorted(os.listdir(seg_path))
    
    num_samples = min(num_samples, len(img_files), len(seg_files))
    
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        img_file = img_files[i]
        seg_file = seg_files[i]
        
        img = Image.open(os.path.join(img_path, img_file))
        seg = Image.open(os.path.join(seg_path, seg_file))
        
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(img)
        plt.title(f"Image: {img_file}")
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(seg, cmap='gray')
        plt.title(f"Mask: {seg_file}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


show_images_and_masks(img_path, seg_path, num_samples=1)