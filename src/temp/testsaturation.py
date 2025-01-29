import cv2
import numpy as np

def check_mask_values(mask_path):
    # Load the mask image in grayscale mode
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Error: Image not found or could not be loaded.")
        return

    # Get the unique values in the mask
    unique_values = np.unique(mask)

    # Print the unique values and what they represent
    print(f"Unique values in the mask: {unique_values}")

    # Optionally, you can interpret these values based on your knowledge of the mask labeling scheme
    for value in unique_values:
        if value == 0:
            print(f"Value {value}: Sky")
        elif value == 1:
            print(f"Value {value}: Clouds")
        else:
            print(f"Value {value}: Other regions (or noise)")

# Path to your mask image
mask_path = "/Users/weijithwimalasiri/Project/data:segmented:thresholded/IMG_1817.PNG"

# Check the mask values
check_mask_values(mask_path)
