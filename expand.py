import cv2

def expand(image_path):
    """
    Expands the size of the input image by twice its width and height.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        np.array: The expanded image.
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found")
    
    new_width = image.shape[1] * 2
    new_height = image.shape[0] * 2
    
    # Resize the image to double the original dimensions
    expanded_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return expanded_image


image_path = 'images/lena.png'
expanded_image = expand(image_path)
cv2.imwrite('expanded_image.jpg',expanded_image) 
