import cv2
import numpy as np
import matplotlib.pyplot as plt

def filter_bright_regions_color(img,filtered_img_save_path, max_brightness_threshold=240):
    """
    Filters out extremely bright regions (e.g., sun) from the color image by turning them black.
    :param img: Input color image (numpy array)
    :param filtered_img_save_path: Path to save the filtered image
    :param max_brightness_threshold: Threshold value to filter bright regions
    :return: Image with bright regions (sun) filtered out
    """
    # Convert the image to grayscale to detect bright regions
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to remove bright regions like the sun in grayscale
    _, bright_mask = cv2.threshold(gray_img, max_brightness_threshold, 255, cv2.THRESH_BINARY)

    # Invert the mask to keep only non-bright regions
    bright_mask_inv = cv2.bitwise_not(bright_mask)

    # Apply the mask to each channel of the original image
    filtered_img = cv2.bitwise_and(img, img, mask=bright_mask_inv)

    # Save the filtered image
    cv2.imwrite(filtered_img_save_path, filtered_img)

    return filtered_img

# Load the image
img = cv2.imread("output/IMG_1816.jpg")
saved = "/data:filtered/IMG_1807.png"

# Step 1: Apply brightness filtering directly on the color image
filtered_color_img = filter_bright_regions_color(img, saved)

# Use matplotlib to display both images
plt.figure(figsize=(10, 5))

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Show the filtered color image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(filtered_color_img, cv2.COLOR_BGR2RGB))
plt.title('Filtered Image (Color)')
plt.axis('off')

# Display the images
plt.tight_layout()
plt.show()
