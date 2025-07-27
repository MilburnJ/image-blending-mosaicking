import cv2
import os
import numpy as np
from expand import expand
from reduce import reduce
from gaussianPyramid import GaussianPyramid

def LaplacianPyramid(image_path, n):
    """
    Builds a Laplacian Pyramid with 'n' levels using the Gaussian Pyramid.
    
    Args:
        image_path (str): Path to the input image.
        n (int): The number of pyramid levels.
    
    Returns:
        list: A list of Laplacian pyramid images.
    """
    # Get the Gaussian Pyramid
    gaussian_pyramid = GaussianPyramid(image_path, n)
    
    laplacian_pyramid = []
    
    # Process each level except the last one
    for i in range(n - 1):
        current_gaussian = cv2.imread(gaussian_pyramid[i])
        
        # Expand the next Gaussian level
        next_gaussian_path = gaussian_pyramid[i + 1]
        expanded_next_gaussian = expand(next_gaussian_path)
        

        
        if expanded_next_gaussian.shape != current_gaussian.shape:
            expanded_next_gaussian = cv2.resize(expanded_next_gaussian, (current_gaussian.shape[1], current_gaussian.shape[0]))

        # Convert both to float32 for subtraction to preserve negative values
        current_gaussian = current_gaussian.astype(np.float32)
        expanded_next_gaussian = expanded_next_gaussian.astype(np.float32)
        
        laplacian = current_gaussian - expanded_next_gaussian
        
        laplacian_pyramid.append(laplacian)
    
    laplacian_pyramid.append(cv2.imread(gaussian_pyramid[-1]).astype(np.float32)) 
    
    return laplacian_pyramid

def save_laplacian_pyramid(laplacian_pyramid):
    """
    Saves the Laplacian Pyramid images after converting them to uint8.
    
    Args:
        laplacian_pyramid (list): List of Laplacian pyramid images.
    """
    for i, laplacian in enumerate(laplacian_pyramid[:-1]):  # Skip the last level
        # Offset by 128 to handle negative values
        laplacian_uint8 = np.clip(laplacian + 128, 0, 255).astype(np.uint8)
        
        laplacian_image_path = f'laplacian_pyramid_level_{i}.jpg'
        cv2.imwrite(laplacian_image_path, laplacian_uint8)
    
    last_level_uint8 = np.clip(laplacian_pyramid[-1], 0, 255).astype(np.uint8)
    cv2.imwrite(f'laplacian_pyramid_level_{len(laplacian_pyramid)-1}.jpg', last_level_uint8)

image_path = 'images/lena.png' 
n_levels = 4 
laplacian_pyramid = LaplacianPyramid(image_path, n_levels)

save_laplacian_pyramid(laplacian_pyramid)

for i in range(n_levels):
    print(f"Laplacian Pyramid Level {i}: laplacian_pyramid_level_{i}.jpg")
