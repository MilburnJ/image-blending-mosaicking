import cv2
import numpy as np

def reduce(image_path):
    """
    Reduces the size of the input image by half its width and height.
    
    Args:
        image_path (str): Path to the input image in RGB format.
    
    Returns:
        np.array: The reduced image.
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found.")
    


    gaussian_blur_x = cv2.GaussianBlur(image, (5, 1), 0)

    gaussian_blur_y = cv2.GaussianBlur(gaussian_blur_x, (1, 5), 0)
    
    # Reduce the size of the image to half its width and height
    reduced_image = cv2.resize(gaussian_blur_y, 
                               (image.shape[1] // 2, image.shape[0] // 2), 
                               interpolation=cv2.INTER_LINEAR)
    
    return reduced_image


image_path = 'images/lena.png'
reduced_image = reduce(image_path)
cv2.imwrite('reduced_image.jpg', reduced_image)
