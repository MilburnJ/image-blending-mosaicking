import os
import tempfile
import cv2
import numpy as np
from expand import expand
from gaussianPyramid import GaussianPyramid
from laplacianPyramid import LaplacianPyramid

# Placeholder for blend region
start_x, end_x = 0, 0

def select_blend_region(event, x, y, flags, param):
    global start_x, end_x
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x = x
    elif event == cv2.EVENT_LBUTTONUP:
        end_x = x

# Function to blend two images using Laplacian pyramid blending
def blend_images(left_image_path, right_image_path, mask, n_levels):
    # Get the Laplacian pyramids for both left and right images
    left_laplacian_pyr = LaplacianPyramid(left_image_path, n_levels)
    right_laplacian_pyr = LaplacianPyramid(right_image_path, n_levels)
    
    # Get the Gaussian pyramid for the mask
    mask_gaussian_pyr = GaussianPyramid(mask, n_levels)

    # Blend each level of the Laplacian pyramid
    blended_pyramid = []
    for l_left, l_right, m in zip(left_laplacian_pyr, right_laplacian_pyr, mask_gaussian_pyr):
        im = cv2.imread(m)
        
        if l_left.shape != l_right.shape:
            l_right = cv2.resize(l_right, (l_left.shape[1], l_left.shape[0]))
        
        if l_left.shape != im.shape:
            im = cv2.resize(im, (l_left.shape[1], l_left.shape[0]))
        blended = l_left * im + l_right * (1 - im)
        blended_pyramid.append(blended)
    
    # Reconstruct the final blended image
    reconstructed_image = blended_pyramid[-1]

    for i in range(n_levels - 2, -1, -1):
        # Save the current reconstructed image to a temporary file
        temp_image_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_image_path = temp_image_file.name
        cv2.imwrite(temp_image_path, reconstructed_image)
        
        temp_image_file.close()
        reconstructed_image = expand(temp_image_path)
        
        # Resize to match the current level if necessary
        if reconstructed_image.shape != blended_pyramid[i].shape:
            reconstructed_image = cv2.resize(reconstructed_image, (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
        
        if reconstructed_image.dtype != blended_pyramid[i].dtype:
            reconstructed_image = reconstructed_image.astype(np.float32)
            blended_pyramid[i] = blended_pyramid[i].astype(np.float32)

        reconstructed_image = cv2.add(reconstructed_image, blended_pyramid[i])
        
        os.remove(temp_image_path)

    return reconstructed_image

if __name__ == "__main__":
    left_image_path = 'images/Test_D1.png' 
    right_image_path = 'images/Test_D2.png' 

    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)
    
    # Ensure images are the same size
    right_image = cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]))

    # Create a window
    cv2.namedWindow('Select Blend Region')
    cv2.setMouseCallback('Select Blend Region', select_blend_region)
    
    while True:
        combined = np.hstack((left_image, right_image))

        # Highlight the blend
        if start_x >= 0 and end_x >= 0:
            overlay = combined.copy()
            cv2.rectangle(overlay, (start_x, 0), (end_x, combined.shape[0]), (0, 255, 0), -1)  # Draw green overlay
            alpha = 0.4
            combined = cv2.addWeighted(overlay, alpha, combined, 1 - alpha, 0)  # Blend the overlay with the image
        
        cv2.imshow('Select Blend Region', combined)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break
    cv2.destroyAllWindows()

    mask = np.zeros_like(left_image[:, :, 0], dtype=np.float32)

    # Ensure that start_x and end_x are within the image boundaries
    start_x = max(0, start_x)
    end_x = min(mask.shape[1], end_x)
    print(f"start_x: {start_x}, end_x: {end_x}, mask width: {mask.shape[1]}")

    # Ensure start_x is less than or equal to end_x
    if start_x > end_x:
        start_x, end_x = end_x, start_x  # Swap values if necessary

    # Generate a 1D blend gradient
    blend_width = end_x - start_x  

    if blend_width > 0:
        # Create a 1D gradient
        blend_gradient = np.linspace(0, 1, blend_width).reshape(1, -1)
        blend_gradient_replicated = np.tile(blend_gradient, (mask.shape[0], 1))  
    
        # Assign the replicated gradient to the mask
        mask[:, start_x:end_x] = blend_gradient_replicated
    else:
        raise ValueError("Invalid blend region")
    mask = np.dstack([mask] * 3)

    # Save the mask as a temp file
    temp_mask_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_mask_path = temp_mask_file.name
    cv2.imwrite(temp_mask_path, mask)


    print(f"Mask saved to: {temp_mask_path}")

    # Blend the images
    n_levels = 6
    blended_image = blend_images(left_image_path, right_image_path, temp_mask_path, n_levels)
    
    # Save the result
    cv2.imwrite('blended_image.jpg', blended_image)
