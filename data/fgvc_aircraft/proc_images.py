"""
Script to find and resize all FGVC-Aircraft images to 84x84.
"""

import numpy as np
import os
from PIL import Image
import glob
import random

np.random.seed(1)
random.seed(2)

def process_and_resize():
    """
    Finds all images in the dataset directory, and resizes them to 84x84.
    """
    # Define the path to the images. The first '*' matches train/val/test folders,
    # the second '*' matches all class folders.
    image_path_pattern = '/home/sv6234/HSML/meta-dataset/FGVC_Aircraft/*/*/*.jpg'
    all_images = glob.glob(image_path_pattern)

    print(f"Found {len(all_images)} images to process.")
    if len(all_images) == 0:
        print("Warning: No images found. Please check the path pattern and directory structure.")
        print(f"Searching in: {image_path_pattern}")
        return

    for i, image_file in enumerate(all_images):
        try:
            # Open the image
            im = Image.open(image_file)
            # Ensure image is RGB
            im = im.convert('RGB')
            # Resize the image
            im = im.resize((84, 84), resample=Image.LANCZOS)
            # Save the image, overwriting the original
            im.save(image_file)

            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1} / {len(all_images)} images.")
        except Exception as e:
            print(f"Could not process {image_file}: {e}")

    print("Finished processing all images.")


if __name__=='__main__':
    process_and_resize()