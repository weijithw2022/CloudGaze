import numpy as np
import cv2
import matplotlib.pyplot as plt

def lucasKanadeAlgorithm(img_path2, img_path1):
    """
    Using the Lucas-Kanade optical flow algorithm, this function computes and visualizes motion between two binary segmented images
    . It also calculates the average 
    motion direction and magnitude.

    Args:
        img_path2 (str): Path to the second segmented image.
        img_path1 (str): Path to the first segmented image.

    Outputs:
        Visualizes motion vectors and prints the average direction and magnitude of motion.
    """

    # Load the binary segmented images as grayscale
    seg_img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    seg_img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    # Ensure the images are binary by applying a threshold
    _, seg_img1 = cv2.threshold(seg_img1, 127, 255, cv2.THRESH_BINARY)
    _, seg_img2 = cv2.threshold(seg_img2, 127, 255, cv2.THRESH_BINARY)

    # Detect feature points (corners) in the first image using Shi-Tomasi corner detection
    p0 = cv2.goodFeaturesToTrack(seg_img1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Define parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), 
                     maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow to track the movement of points from the first image to the second
    p1, st, err = cv2.calcOpticalFlowPyrLK(seg_img1, seg_img2, p0, None, **lk_params)

    # Filter out successfully tracked points
    good_new = p1[st == 1]  # Points in the second image
    good_old = p0[st == 1]  # Corresponding points in the first image

    # Create an RGB visualization image to display motion vectors
    flow_visualization = np.dstack((seg_img2, seg_img2, seg_img2))

    # Initialize a list to store motion vectors (dx, dy) for calculating average motion
    motion_vectors = []

    # Draw motion vectors and calculate direction for each pair of points
    for new, old in zip(good_new, good_old):
        a, b = map(int, new.ravel())  # Coordinates of the new point
        c, d = map(int, old.ravel())  # Coordinates of the old point

        # Calculate the motion vector components (dx, dy)
        dx = c - a
        dy = d - b

        motion_vectors.append((dx, dy))  # Store the motion vector for averaging

    # Calculate and display the average motion vector if any vectors are found
    if motion_vectors:
        avg_dx = np.mean([v[0] for v in motion_vectors])  # Average x component of motion
        avg_dy = np.mean([v[1] for v in motion_vectors])  # Average y component of motion

        # Calculate the angle and magnitude of the average motion vector
        angle = np.arctan2(avg_dy, avg_dx) * 180 / np.pi  # Direction in degrees
        magnitude = np.sqrt(avg_dx**2 + avg_dy**2)  # Magnitude of the motion

        # Print the average motion direction and magnitude
        print(f"Average motion direction: {angle:.2f} degrees, Magnitude: {magnitude:.2f}")
        print(f"Average dx: {avg_dx:.2f}, Average dy: {avg_dy:.2f}")

        # Draw the mean motion vector on the visualization image
        img_center = (1855, 1563)  # Replace with the center coordinates of your image
        mean_endpoint = (int(1855 + avg_dx * 10), int(1563 + avg_dy * 10))  # Scaled vector

        # Draw the mean motion vector as a red arrow
        flow_visualization = cv2.arrowedLine(flow_visualization, img_center, mean_endpoint, (255, 0, 0), 3)

    # Display the flow visualization using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(flow_visualization)
    plt.title('Flow Visualization with Mean Motion Vector')
    plt.show()


# Example usage: Paths to binary segmented images
img_path_1 = "/data:segmented:thresholded/IMG_1816.PNG"
img_path_2 = "/Project/data:segmented:thresholded/IMG_1817.PNG"

# Call the function
lucasKanadeAlgorithm(img_path_1, img_path_2)
