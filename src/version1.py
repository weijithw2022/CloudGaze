import cv2
import numpy as np
import math
from kalmanfilter import KalmanFilter
import time
import os

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

dt = 1.0
process_noise = np.eye(4) * 0.1
measurement_noise = np.eye(2) * 0.5
initial_state = np.array([0, 0, 0, 0])
initial_covariance = np.eye(4)

def filter_sun_get_centroid(img, filtered_img_save_path, max_brightness_threshold=240):
    """
    Filters out extremely bright regions (e.g., sun) from the color image by turning them black.
    :param img: Input color image (numpy array)
    :param max_brightness_threshold: Threshold value to filter bright regions
    :param filtered_img_save_path: Path to save the filtered image
    :return: cX, cY centroids of the sun
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # All the pixels that have a value greater than max_brightness_threshold will be classified as bright
    _, bright_mask = cv2.threshold(gray_img, max_brightness_threshold, 255, cv2.THRESH_BINARY)

    if np.any(bright_mask):
        # Invert the bright mask to keep only non-bright regions
        bright_mask_inv = cv2.bitwise_not(bright_mask)

        # Apply the mask to each channel of the original image
        filtered_img = cv2.bitwise_and(img, img, mask=bright_mask_inv)

        # Save the filtered image
        cv2.imwrite(filtered_img_save_path, filtered_img)

        # Find contours of the bright regions
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour, which should correspond to the sun
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the moments of the largest contour
        M = cv2.moments(largest_contour)

        # Get the centroid of the sun
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0  # If the contour has zero area

        # Draw a circle around the sun
        radius = int(cv2.contourArea(largest_contour) ** 0.5)  # Estimate the radius from the area of the contour

        return (cX, cY)

    else:
        print("No bright regions detected.")
        return None

def segment_clouds_and_sky(filtered_img_path , cloud_thresholds, sky_thresholds, save_mask_path):
    """
        Segments clouds and sky regions from a filtered image using specified color thresholds and saves the
        resulting mask.

        :param filtered_img_path: Path to the input filtered image (numpy array)
        :param cloud_thresholds: List of tuples defining the HSV color thresholds for identifying clouds.
        :param sky_thresholds: List of tuples defining the HSV color thresholds for identifying the sky.
        :param save_mask_path: Path where the resulting cloud mask will be saved after processing.

        :return: None (the function saves the cloud mask to the specified path).
    """
    # Read the image in BGR format and convert it directly to HSV color space
    hsv = cv2.cvtColor(cv2.imread(filtered_img_path), cv2.COLOR_BGR2HSV)

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

    # Apply morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)

    # Save the cloud mask if a save path is provided
    cv2.imwrite(save_mask_path, cloud_mask)

    print("Saving is done.")

def find_largest_cloud_centroid(mask_img_path):
    """
    Finds the centroids of the clouds in a binary segmented image.

    :param mask_img_path: Path to the input mask image where the cloud is white (255) and sky is black (0).
    :return: A tuple (cx, cy) representing the centroid of the largest cloud, or (None, None) if no cloud is found.
    """
    # Load the binary segmented image
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    # Making sure the clouds are 255 and sky is 0
    _, mask_img = cv2.threshold(mask_img, 150, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image (clouds are white, sky is black)
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours found!")
        return None

    # Take the largest contour assuming it's the main cloud
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the moments of the contour
    M = cv2.moments(largest_contour)

    if M["m00"] != 0:  # Avoid division by zero
        # Calculate the centroid using moments
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy)
    else:
        return None

def lucasKanadeAlgorithm(mask_img_path_2, mask_img_path_1):
    """
    Computes the optical flow between two masked images using the Lucas-Kanade method and
    returns the average motion direction (angle), magnitude, and average displacement components (dx, dy).

    :param mask_img_path_2: Path to the second segmented image (current frame).
    :param mask_img_path_1: Path to the first segmented image (previous frame).
    :return: angle (degrees), magnitude (pixels), avg_dx, avg_dy representing average motion in x and y.
    """

    seg_img1 = cv2.imread(mask_img_path_1, cv2.IMREAD_GRAYSCALE)
    seg_img2 = cv2.imread(mask_img_path_2, cv2.IMREAD_GRAYSCALE)

    # Ensure the images are binary
    _, seg_img1 = cv2.threshold(seg_img1, 127, 255, cv2.THRESH_BINARY)
    _, seg_img2 = cv2.threshold(seg_img2, 127, 255, cv2.THRESH_BINARY)

    # Detect points to track using Shi-Tomasi corner detection or all non-zero points
    p0 = cv2.goodFeaturesToTrack(seg_img1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    # Calculate optical flow between the two segmented images
    p1, st, err = cv2.calcOpticalFlowPyrLK(seg_img1, seg_img2, p0, None, **lk_params)

    # Select the points that were successfully tracked
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Initialize variables to calculate the average movement direction
    motion_vectors = []

    # Draw the motion vectors and calculate direction
    for new, old in zip(good_new, good_old):
        a, b = map(int, new.ravel())
        c, d = map(int, old.ravel())

        # Calculate the motion vector (dx, dy)
        dx = c-a
        dy = d-b

        motion_vectors.append((dx, dy))

    if motion_vectors:

        # Get the mean of dx and dy in motion vectors
        avg_dx = np.mean([v[0] for v in motion_vectors])
        avg_dy = np.mean([v[1] for v in motion_vectors])

        angle = np.arctan2(avg_dy, avg_dx)*180/np.pi
        magnitude = np.sqrt(avg_dx**2 + avg_dy**2)

        return (angle, magnitude, avg_dx, avg_dy)

    return None

def get_prob_cloud_cover(sun_centroid, largest_cloud_centroid, avg_dx, avg_dy):
    """
    Calculate the probability of cloud coverage based on the motion of the largest cloud.

    :param sun_centroid: Tuple (cX, cY) representing the coordinates of the sun's centroid.
    :param largest_cloud_centroid: Tuple (lX, lY) representing the coordinates of the largest cloud's centroid.
    :param avg_dx: Average displacement in the x-direction for the cloud.
    :param avg_dy: Average displacement in the y-direction for the cloud.
    :return: Probability (in percentage) that the cloud will cover the sun.
    """
    # Sun's centroid
    cX, cY = sun_centroid
    # Largest_cloud_centroid
    lX, lY = largest_cloud_centroid
    # Ceil the avg_dx and avg_dy values
    ceil_avg_dx = math.ceil(avg_dx)
    ceil_avg_dy = math.ceil(avg_dy)

    # Distance from sun's centroid to the new largest cloud's centroid
    d1 = np.sqrt((lX-cX)**2 + (lY-cY)**2)

    # Distance from sun's centroid to the predicted motion of the largest_cloud_centroid
    pX = lX + ceil_avg_dx
    pY = lY + ceil_avg_dy
    d2 = np.sqrt((pX-cX)**2 + (pY-cY)**2)

    # Chance of cloud's covering the sun
    chance = (d1/d2)*100
    return chance

def get_distance(sun_centroid, largest_cloud_centroid):
    """
    Calculate the probability of cloud coverage based on the motion of the largest cloud.

    :param sun_centroid: Tuple (cX, cY) representing the coordinates of the sun's centroid.
    :param largest_cloud_centroid: Tuple (lX, lY) representing the coordinates of the largest cloud's centroid.
    :param avg_dx: Average displacement in the x-direction for the cloud.
    :param avg_dy: Average displacement in the y-direction for the cloud.
    :return: Probability (in percentage) that the cloud will cover the sun.
    """
    # Sun's centroid
    cX, cY = sun_centroid
    # Largest_cloud_centroid
    lX, lY = largest_cloud_centroid
    # Distance from sun's centroid to the new largest cloud's centroid
    distance = np.sqrt((lX-cX)**2 + (lY-cY)**2)

    return distance

def main():
    folder_path = ""
    saved_path = ""
    segmented_path = ""
    distances = []
    kf = KalmanFilter(dt=dt,
                      process_noise=process_noise,
                      measurement_noise=measurement_noise,
                      initial_state=initial_state,
                      initial_covariance=initial_covariance)
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for file in image_files:
        if file is None:
            print(f"Failed to load {file}")
            continue

        # Load the image
        img = cv2.imread(file)

        file_name = os.path.splitext(file)[0] + '.png'
        # Save path for the sun filtered image
        saved = os.path.join(saved_path, file_name)

        # Apply the filter and get the sun's centroid
        sun_centroid = filter_sun_get_centroid(img, saved)

        if sun_centroid:
            cX, cY = sun_centroid
            print(f"Centroid of the bright region (sun) is at ({cX}, {cY})")
        else:
            print("No sun detected.")

        segmented_file_path = os.path.join(segmented_path, file_name)
        segment_clouds_and_sky(saved, cloud_thresholds, sky_thresholds, segmented_file_path)

        largest_cloud_centroid = find_largest_cloud_centroid(segmented_file_path)

        if largest_cloud_centroid:
            cx, cy = largest_cloud_centroid
            print(f"Largest cloud centroid: ({cx}, {cy})")
        else:
            print("No cloud found or no valid centroid.")

        # Get the distance between Sun's centroid and Largest Cloud's centroid
        distance = get_distance(sun_centroid, largest_cloud_centroid)
        distances.append(distance)

        # Apply Kalman Filter
        kf.predict()
        kf.update(np.array([cx, cy]))

        current_file_name = os.path.basename(segmented_file_path)
        # Extract the numerical part of the file name
        prefix, number, extension = current_file_name.split('_')[0], current_file_name.split('_')[1].split('.')[0], current_file_name.split('.')[1]

        # Calculate the previous file number
        previous_number = int(number) - 1
        previous_file_name = f"IMG_{previous_number:04d}.{extension}"  # Ensure 4 digits in number

        # Construct the path to the previous file
        previous_file_path = os.path.join(segmented_path, previous_file_name)

        # Check if the previous file exists
        if os.path.exists(previous_file_path):
            # Call the Lucas-Kanade algorithm function
            result = lucasKanadeAlgorithm(previous_file_path, segmented_file_path)
            # Check if result is not None, then print the returned values
            if result:
                angle, magnitude, avg_dx, avg_dy = result
                print(f"Angle: {angle:.2f} degrees")
                print(f"Magnitude: {magnitude:.2f} pixels")
                print(f"Average dx: {avg_dx:.2f}")
                print(f"Average dy: {avg_dy:.2f}")
            else:
                print("No motion vectors found or insufficient points to calculate flow.")

            if distances[-1]> distance or distances[-1] == distance:
                state = kf.get_state()
                x, y, vx, vy = state
                distance = np.sqrt((cX - x) ** 2 + (cY - y) ** 2)
                avg_v = np.sqrt(vx ** 2 + vy ** 2)
                predicted_time = distance/avg_v

                print("Send an email.")

        else:
            print("No previous file exists. Current file is the only one in the folder.")















