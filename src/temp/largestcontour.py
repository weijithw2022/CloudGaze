import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_cloud_centroids(seg_img_path):
    """
    Finds the centroids of the clouds in a binary segmented image.

    :param seg_img_path: Path to the input binary image where the cloud is white (255) and sky is black (0).
    :return: A list of tuples (cX, cY) representing the centroids of the cloud.
    """
    # Load the binary segmented image
    seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)

    _, seg_img = cv2.threshold(seg_img, 150, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image (clouds are white, sky is black)
    contours, _ = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours found!")
        return None, None

    # Convert the grayscale image to BGR so we can draw colored circles
    color_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)

    # Take the largest contour assuming it's the main cloud
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the moments of the contour
    M = cv2.moments(largest_contour)

    if M["m00"] != 0:  # Avoid division by zero
        # Calculate the centroid using moments
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Print the centroid coordinates
        print(f"Centroid of the largest contour: ({cx}, {cy})")

        # Draw the contour and centroid for debugging purposes
        cv2.drawContours(color_img, [largest_contour], -1, (0, 255, 0), 20)  # Draw the contour in gray

        # Draw the centroid as a blue circle
        cv2.circle(color_img, (cx, cy), 50, (255, 0, 0), -1)  # Blue circle for centroids


    return color_img


# Test with your segmented image path
seg_img_path = "/Users/weijithwimalasiri/Project/data:segmented:thresholded/IMG_1816.PNG"  # The image you uploaded
result_img = find_cloud_centroids(seg_img_path)

# Show the image with marked centroids using matplotlib
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title('Cloud Centroids Marked')
plt.axis('off')
plt.show()

