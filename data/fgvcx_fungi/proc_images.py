"""
Script to find and resize all FGVCx Fungi images to 84x84.
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
    # Define the base path to search in
    search_path = '/home/sv6234/HSML/meta-dataset/FGVCx_Fungi/'
    # Create recursive patterns for both lowercase and uppercase extensions
    image_path_pattern_lower = os.path.join(search_path, '**', '*.jpg')
    image_path_pattern_upper = os.path.join(search_path, '**', '*.JPG')

    # Find all images with both extensions and combine the lists
    all_images_lower = glob.glob(image_path_pattern_lower, recursive=True)
    all_images_upper = glob.glob(image_path_pattern_upper, recursive=True)
    all_images = all_images_lower + all_images_upper

    print(f"Found {len(all_images)} images to process.")
    if len(all_images) == 0:
        print("Warning: No images found. Please check the path pattern and directory structure.")
        print(f"Searching recursively in: {search_path}")
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


if __name__ == '__main__':
    process_and_resize()
