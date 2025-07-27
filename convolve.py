import cv2
import numpy as np

def convolve(image_path, kernel):
    """
    Convolves an input image (in grayscale format) with a given kernel manually.
    
    Args:
        image_path (str): Path to the input image.
        kernel (np.array): The convolution kernel (K).
    
    Returns:
        np.array: The manually convolved output image.
    """
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)  # Convert to float32 for precision
    
    if image is None:
        raise ValueError("Image not found.")
    
    # Normalize kernel if sum not zero
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel = kernel / kernel_sum
    

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    

    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    
    # Pad the image using 'reflect' padding to mimic cv2.filter2D()
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_REFLECT)
    
    output_image = np.zeros_like(image)
    
    #convolution
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            
            output_value = np.sum(region * kernel)
            output_image[i, j] = output_value
    
    # Clip values to be in the valid range [0, 255]
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image


kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X-Direction Kernel (Sobel)
image_path = 'images/lena.png' 

output_x = convolve(image_path, kernel_x)
cv2.imwrite('convolved_image_x.jpg', output_x)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
convolved_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel_x)
cv2.imwrite("test_convolved_image_x.jpg",convolved_image)
