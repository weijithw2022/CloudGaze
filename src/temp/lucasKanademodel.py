import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import queue
import sys
import os
sys.path.append(os.path.abspath("/src/models"))
from loss import *
import time

# Load the trained TensorFlow model with custom loss functions for segmentation tasks
model = tf.keras.models.load_model('/weights/ACLNet_Best.keras',
                                   custom_objects={
                                       'diceCoef': diceCoef,
                                       'bceDiceLoss': bceDiceLoss
                                   })


def getsegmented_image(img_path):
    """
    Segments an input image using a pre-trained model and saves the binary segmented result.
    Steps:
    
    Args:
        img_path (str): Path to the input image.

    Returns:
        str: Path to the saved binary segmented image.
    """
  
    img1 = Image.open(img_path).resize((288, 288))  # Preprocess: resize and normalize image
    img1_array = np.array(img1).astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    img1_array = np.expand_dims(img1_array, axis=0)  # Add batch dimension

    input_data = [img1_array, img1_array]  # Prepare input for the model
    output_data = model.predict(input_data)  # Perform inference

    # Extract the segmentation mask and convert it to a binary mask
    segmentation_mask = np.argmax(output_data[0], axis=-1)
    binary_segmentation = np.where(segmentation_mask > 0, 255, 0).astype(np.uint8)

    # Save the binary segmentation result
    path = "/Users/weijithwimalasiri/Project/data:segmented"
    filename = os.path.basename(img_path)
    target_path = os.path.join(path, filename)
    cv2.imwrite(target_path, binary_segmentation)

    return target_path


def lucasKanadeAlgorithm(img_path2, img_path1):
    """
    Computes and visualizes motion between two consecutive binary segmented images
    using the Lucas-Kanade optical flow algorithm.

    Args:
        img_path2 (str): Path to the second segmented image.
        img_path1 (str): Path to the first segmented image.
    """
    # Load the binary segmented images as grayscale
    seg_img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    seg_img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    # Perform Shi-Tomasi corner detection to identify key points to track
    p0 = cv2.goodFeaturesToTrack(seg_img1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Set parameters for the Lucas-Kanade optical flow algorithm
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Compute optical flow between the two images
    p1, st, err = cv2.calcOpticalFlowPyrLK(seg_img1, seg_img2, p0, None, **lk_params)

    # Visualize the optical flow by drawing motion vectors
    flow_visualization = np.dstack((seg_img2, seg_img2, seg_img2))
    for i, (new, old) in enumerate(zip(p1[st == 1], p0[st == 1])):
        a, b = map(int, new.ravel())
        c, d = map(int, old.ravel())
        flow_visualization = cv2.arrowedLine(flow_visualization, (a, b), (c, d), (0, 255, 0), 2)

    # Save the flow visualization to a specified directory
    path = "/Users/weijithwimalasiri/Project/data:followed"
    filename = os.path.basename(img_path2)
    target_path = os.path.join(path, filename)
    cv2.imwrite(target_path, flow_visualization)


def lucas_kanade_method(video_path):
    """
    Applies the Lucas-Kanade optical flow method to track motion in a video.
    Steps:
    1. Load video frames sequentially.
    2. Detect corners in the first frame.
    3. Track these corners across frames using the optical flow method.
    4. Visualize the motion vectors on the video frames.

    Args:
        video_path (str): Path to the input video.
    """
    cap = cv2.VideoCapture(video_path)  # Load the video

    # Parameters for detecting corners and computing optical flow
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Process the video frames
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # Detect corners
    mask = np.zeros_like(old_frame)  # Initialize mask for visualization


if __name__ == '__main__':
    """
    Main script for processing a queue of images:
    1. Segment each image and save the binary segmentation results.
    2. Compute and visualize motion between consecutive segmented images.
    3. Simulate a 30-second wait before processing the next pair of images.
    """
    img_queue = queue.Queue()  # Queue for storing input image paths
    process_queue = queue.Queue()  # Queue for storing segmented image paths

    img_paths = [
        "/data:filtered/IMG_1816.png",
        "/data:filtered/IMG_1818.png"
    ]

    # Populate the image queue with paths
    for img_path in img_paths:
        img_queue.put(img_path)

    # Process the images in the queue
    while not img_queue.empty():
        current_img_path = img_queue.get()
        segmented_img_path = getsegmented_image(current_img_path)  # Segment the image
        process_queue.put(segmented_img_path)

        if process_queue.qsize() == 2:
            img_path_1 = process_queue.get()
            img_path_2 = process_queue.queue[0]
            lucasKanadeAlgorithm(img_path_1, img_path_2)  # Compute optical flow
        else:
            print(f"Queue has {process_queue.qsize()} images.")

        time.sleep(30)  # Simulate a 30-second delay
