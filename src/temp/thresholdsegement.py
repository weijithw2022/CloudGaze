import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_clouds_and_sky(image_path, cloud_thresholds, sky_thresholds, save_mask_path):

    # Convert BGR to RGB and then to HSV for processing
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #hsv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2HSV)

    # Initialize masks for sky and clouds
    sky_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cloud_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Step 1: Apply sky thresholds
    for threshold in sky_thresholds:
        lower_bound = np.array([threshold[0], threshold[2], threshold[4]])  # Hue, Saturation, Value
        upper_bound = np.array([threshold[1], threshold[3], threshold[5]])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        sky_mask = cv2.bitwise_or(sky_mask, mask)

    # Step 2: Apply cloud thresholds only on areas that are NOT sky
    non_sky_area = cv2.bitwise_not(sky_mask)  # Invert sky mask to get non-sky regions

    for threshold in cloud_thresholds:
        lower_bound = np.array([threshold[0], threshold[2], 50])  # Adjusted for refined saturation (Hue, Saturation, Value)
        upper_bound = np.array([threshold[1], threshold[3], 255]) # Adjusted saturation limits
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.bitwise_and(mask, non_sky_area)  # Only apply cloud mask to non-sky areas
        cloud_mask = cv2.bitwise_or(cloud_mask, mask)

    # Combine the masks for clouds and sky
    #final_mask = cv2.bitwise_or(cloud_mask, sky_mask)

    # Apply morphological operations to clean the mask
    #kernel = np.ones((5, 5), np.uint8)
    #final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    #final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    # Extract clouds and sky regions from the original image
    #segmented_img = cv2.bitwise_and(img, img, mask=final_mask)

    # Apply morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)

    # Save the cloud mask if a save path is provided
    cv2.imwrite(save_mask_path, cloud_mask)

    # Display the results
    #plt.figure(figsize=(12, 6))
    #plt.subplot(1, 3, 1)
    #plt.title("Original Image")
    #plt.imshow(img)
    #plt.axis('off')

    #plt.subplot(1, 3, 2)
    #plt.title("Cloud Mask")
    #plt.imshow(cloud_mask, cmap='gray')
    #plt.axis('off')

    #plt.subplot(1, 3, 3)
    #plt.title("Sky Mask")
    #plt.imshow(sky_mask, cmap='gray')
    #plt.axis('off')

    #plt.show()

# Example usage
image_path = "/data:filtered/IMG_1807.png"
save_mask_path = "/data:segmented/IMG_1807.png"

# Define thresholds for clouds and sky
cloud_thresholds = [
    (0, 80, 50, 255),  # Low saturation clouds (gray and white)
    (180, 255, 50, 255),  # Very bright clouds
    (100, 115, 20, 70),  # Unsegmented cloud areas based on new values
]

sky_thresholds = [
    (90, 100, 50, 100, 50, 180),   # Light blue sky (low saturation and brightness)
    (120, 160, 50, 100, 50, 180),  # Darker sky (deeper blue, avoiding overlap with clouds)
]

# Call the function with the image path and save the cloud mask
image_path = "/data:filtered/IMG_1807.png"
save_mask_path = "/data:segmented:thresholded/IMG_1807_1.PNG"
segment_clouds_and_sky(image_path, cloud_thresholds, sky_thresholds, save_mask_path)
