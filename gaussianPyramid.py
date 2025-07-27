import cv2
import os
from reduce import reduce

def GaussianPyramid(image_path, n):
    """
    Builds a Gaussian Pyramid with 'n' levels.
    
    Args:
        image_path (str): Path to the input image.
        n (int): The number of pyramid levels.
    
    Returns:
        list: A list of image paths, where each image is a level in the Gaussian Pyramid.
    """
    pyramid_image_paths = []
    
    # Save the original image as the first level
    print(image_path)
    current_image = cv2.imread(image_path)
    original_image_path = f'gaussian_pyramid_level_0.jpg'
    cv2.imwrite(original_image_path, current_image)
    pyramid_image_paths.append(original_image_path)
    
    # Iteratively apply the reduce function
    current_image_path = image_path
    for i in range(1, n):
        reduced_image = reduce(current_image_path)
        
        reduced_image_path = f'gaussian_pyramid_level_{i}.jpg'
        cv2.imwrite(reduced_image_path, reduced_image)
        
        pyramid_image_paths.append(reduced_image_path)
        
        current_image_path = reduced_image_path
    
    return pyramid_image_paths


image_path = 'images/lena.png' 
n_levels = 4 
pyramid = GaussianPyramid(image_path, n_levels)

for i, path in enumerate(pyramid):
    print(f"Pyramid Level {i}: {path}")
