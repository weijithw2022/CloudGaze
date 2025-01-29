import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_sun_position(image_path, max_brightness_threshold=240):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to create a binary mask of bright regions
    _, bright_mask = cv2.threshold(gray_img, max_brightness_threshold, 255, cv2.THRESH_BINARY)

    # Invert the mask to keep only non-bright regions
    bright_mask_inv = cv2.bitwise_not(bright_mask)

    # Apply the mask to each channel of the original image
    filtered_img = cv2.bitwise_and(img, img, mask=bright_mask_inv)

    # Find contours of the bright regions
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found
    if contours:
        # Find the largest contour, which should correspond to the sun
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the moments of the largest contour
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            # Calculate the centroid of the contour
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0  # If the contour has zero area

        # Set a small fixed radius to draw a circle around the centroid (instead of covering the sun)
        fixed_radius = 10  # Adjust this value as needed for the size of the circle around the centroid

        # Draw a small red circle at the centroid
        cv2.circle(filtered_img, (cX, cY), fixed_radius, (0, 0, 255), -1)  # Red color (BGR: 0,0,255)

        # Show the original image with the sun position
        plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
        plt.title('Sun Position with Circle')
        plt.axis('off')
        plt.show()

        return (cX, cY)  # Return the position of the sun as (x, y)

    else:
        print("No bright regions found.")
        return None


# Example usage
image_path = "/output/IMG_1817.jpg"
sun_position = find_sun_position(image_path)
if sun_position:
    print(f"The position of the sun is at: {sun_position}")
