import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_mask(mask_path):
    # Read the mask as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the mask has binary values (0 for sky, 255 for clouds/noise)
    # Create a binary mask where 255 remains 255 (clouds) and 0 stays 0 (sky)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Visualize the binary mask to confirm it's correct
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Binary Mask (Cloud Regions)")
    plt.axis('off')
    plt.show()

    # Extract the coordinates of the 255 (cloud) regions for tracking
    points_to_track = cv2.goodFeaturesToTrack(binary_mask, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    print(f"Number of cloud points detected: {len(points_to_track) if points_to_track is not None else 0}")


# Path to your mask image
mask_path = "/Users/weijithwimalasiri/Project/data:segmented/0039.png"

# Check the mask values
preprocess_mask(mask_path)
