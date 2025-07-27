import cv2
import numpy as np
from expand import expand

def Reconstruct(LI, n):
    """
    Reconstructs the original image by collapsing the Laplacian pyramid.
    
    Args:
        LI (list): A list of file paths representing the Laplacian pyramid images.
        n (int): The number of pyramid levels.
    
    Returns:
        np.array: The reconstructed image.
    """
    current_image = cv2.imread(LI[-1])
    if current_image is None:
        raise ValueError("Failed to load the smallest Gaussian image.")
    
    print(f"Starting reconstruction with smallest image size: {current_image.shape[:2]}")
    
    for i in range(n - 2, -1, -1):  # Go from the second-to-last level down to the first
        expanded_image = expand(LI[i + 1])  # Expand the image from the next level
        
        if expanded_image is None:
            raise ValueError(f"Failed to expand image for level {i+1}.")
        

        
        laplacian_level = cv2.imread(LI[i])
        if laplacian_level is None:
            raise ValueError(f"Failed to load Laplacian level {i}.")
        
        # Convert both images to float32 for the addition
        laplacian_level = laplacian_level.astype(np.float32)
        expanded_image = expanded_image.astype(np.float32)
        
        
        # Add the expanded image to the current Laplacian level
        current_image = laplacian_level + expanded_image
        
        # Clip values to be in the valid range
        current_image = np.clip(current_image, 0, 255).astype(np.uint8)
    
    print(f"Reconstructed image size: {current_image.shape[:2]}")
    return current_image

laplacian_pyramid = [
    'laplacian_pyramid_level_0.jpg',
    'laplacian_pyramid_level_1.jpg',
    'laplacian_pyramid_level_2.jpg',
    'gaussian_pyramid_level_3.jpg'  #smallest Gaussian level
]

# Reconstruct the original image
reconstructed_image = Reconstruct(laplacian_pyramid, len(laplacian_pyramid))

# Save the reconstructed image
cv2.imwrite('reconstructed_image.jpg', reconstructed_image)

# Compute reconstruction error
def compute_reconstruction_error(original_image_path, reconstructed_image):
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError("Failed to load the original image for error computation.")
    
    difference = cv2.absdiff(original_image, reconstructed_image)
    error = np.sum(difference)
    
    return error

original_image_path = 'images/lena.png'
reconstruction_error = compute_reconstruction_error(original_image_path, reconstructed_image)

print(f'Reconstruction error: {reconstruction_error}')