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

    # Invert the binary image so that the clouds become black (foreground) and the sky becomes white (background)
    seg_img_inverted = cv2.bitwise_not(seg_img)

    # Find contours in the binary image (clouds are white, sky is black)
    contours, _ = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the grayscale image to BGR so we can draw colored circles
    color_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)

    # List to store centroids
    centroids = []

    for contour in contours:
        # Calculate the moments of the contour
        M = cv2.moments(contour)

        if M["m00"] != 0:  # Avoid division by zero
            # Calculate the centroid using moments
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

            # Draw the contour and centroid for debugging purposes
            cv2.drawContours(color_img, [contour], -1, (127, 127, 127), 2)  # Draw the contour in gray

            # Draw the centroid as a blue circle
            cv2.circle(color_img, (cx, cy), 7, (255, 0, 0), -1)  # Blue circle for centroids

    return centroids, color_img


# Test with your segmented image path
seg_img_path = "/data:segmented/0039.png"  # The image you uploaded
centroids, result_img = find_cloud_centroids(seg_img_path)

# Print centroids
print("Cloud centroids:", centroids)

# Show the image with marked centroids using matplotlib
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title('Cloud Centroids Marked')
plt.axis('off')
plt.show()

