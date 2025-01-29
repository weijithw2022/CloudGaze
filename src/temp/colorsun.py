import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_and_color_sun(image_path, max_brightness_threshold=240):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to create a binary mask of bright regions
    _, bright_mask = cv2.threshold(gray_img, max_brightness_threshold, 255, cv2.THRESH_BINARY)

    # Find contours of the bright regions
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found
    if contours:
        # Find the largest contour, which should correspond to the sun
        largest_contour = max(contours, key=cv2.contourArea)

        # Fill the area of the sun with red color
        cv2.drawContours(img, [largest_contour], -1, (0, 0, 255), thickness=cv2.FILLED)

        # Show the image with the sun colored red
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Sun Colored in Red')
        plt.axis('off')
        plt.show()

        # Return the modified image
        return img

    else:
        print("No bright regions found.")
        return None

# Example usage
image_path = "/Users/weijithwimalasiri/Desktop/UoM_S5/CS3283_Embedded_Systems_Project/Project/images/output/IMG_1807.jpg"
colored_image = find_and_color_sun(image_path)
