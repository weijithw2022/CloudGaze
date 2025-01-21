import numpy as np
import cv2
import matplotlib.pyplot as plt
import queue
import sys
import os
sys.path.append(os.path.abspath("/Users/weijithwimalasiri/Project/src/models"))

def lucasKanadeAlgorithm(img_path1, img_path2):

    # Read the binary segmented images as grayscale
    seg_img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    seg_img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    # Ensure the images are binary (optional, in case they aren't perfect binary)
    _, seg_img1 = cv2.threshold(seg_img1, 127, 255, cv2.THRESH_BINARY)
    _, seg_img2 = cv2.threshold(seg_img2, 127, 255, cv2.THRESH_BINARY)

    # Detect points to track (using Shi-Tomasi corner detection or all non-zero points)
    p0 = cv2.goodFeaturesToTrack(seg_img1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow between the two segmented images
    p1, st, err = cv2.calcOpticalFlowPyrLK(seg_img1, seg_img2, p0, None, **lk_params)

    # Select the points that were successfully tracked
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Create an RGB image to visualize the flow
    flow_visualization = np.dstack((seg_img2, seg_img2, seg_img2))

    # Draw the motion vectors on the flow visualization
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = map(int, new.ravel())  # Ensure (a, b) is a tuple of integers
        c, d = map(int, old.ravel())  # Ensure (c, d) is a tuple of integers

        # Debugging: Print the types and values of coordinates
        print(f"New Point: ({a}, {b}), Old Point: ({c}, {d})")
        print(f"Type of new: {type(a)}, {type(b)}; Type of old: {type(c)}, {type(d)}")

        # Check if points are within image boundaries
        if 0 <= a < flow_visualization.shape[1] and 0 <= b < flow_visualization.shape[0] and \
           0 <= c < flow_visualization.shape[1] and 0 <= d < flow_visualization.shape[0]:
            flow_visualization = cv2.arrowedLine(flow_visualization, (a, b), (c, d), (0, 255, 0), 2)

    path = "/Users/weijithwimalasiri/Project/data:followed"
    filename = os.path.basename(img_path2)
    target_path = os.path.join(path, filename)
    cv2.imwrite(target_path, flow_visualization)

    # Display the original second image and the flow visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(seg_img1, cmap='gray')
    plt.title("First Binary Segmented Image")

    plt.subplot(1, 2, 2)
    plt.imshow(flow_visualization)
    plt.title("Optical Flow Visualization")
    path_name = "/Users/weijithwimalasiri/Project/data:predicted"
    target = os.path.join(path_name, filename)
    plt.savefig(target)


if __name__ == '__main__':
    img_queue = queue.Queue()
    process_queue = queue.Queue()

    img_path_1 = "/Users/weijithwimalasiri/Project/data:segmented:thresholded/IMG_1816.PNG"
    img_path_2 = "/Users/weijithwimalasiri/Project/data:segmented:thresholded/IMG_1818.PNG"

    lucasKanadeAlgorithm(img_path_1, img_path_2)









