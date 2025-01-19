import cv2
import matplotlib.pyplot as plt


def find_all_cloud_centroids(seg_img_path):
    """
    Finds the centroids of all clouds in a masked image where clouds are represented by 255.

    :param seg_img_path: Path to the input masked image where the cloud is 255.
    :return: The image with all contours drawn and centroids marked, and a list of centroid coordinates.
    """
    # Load the masked image (cloud = 255, sky = 0)
    seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)

    _, seg_img = cv2.threshold(seg_img, 150, 255, cv2.THRESH_BINARY)

    # Find contours in the image where clouds are 255
    contours, _ = cv2.findContours(seg_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours found!")
        return None, []

    # Convert the grayscale image to BGR for colored drawing
    color_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)

    # Initialize a list to store the centroids of all contours
    centroids = []

    # Loop through all contours
    for contour in contours:
        # Calculate the moments of the contour
        M = cv2.moments(contour)

        if M["m00"] != 0:  # Avoid division by zero
            # Calculate the centroid using moments
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Store the centroid in the list
            centroids.append((cx, cy))

            # Draw the contour on the image
            cv2.drawContours(color_img, [contour], -1, (255, 165, 0), 20, lineType=cv2.LINE_AA)

            # Draw the centroid as a blue circle
            #cv2.circle(color_img, (cx, cy), 50, (255, 0, 0), -1)  # Blue circle for the centroid

    # Print all centroids for debugging
    print("Centroids of all contours:", centroids)

    return color_img, centroids


# Example usage
seg_img_path = "data:segmented:thresholded/IMG_1816.PNG"
output_image, centroids = find_all_cloud_centroids(seg_img_path)

if output_image is not None:
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Cloud Contours and Centroids")
    plt.show()
