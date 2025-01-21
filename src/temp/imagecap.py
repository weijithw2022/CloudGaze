import cv2

def create_capture(source=0):
  """
    Create a video capture object for a given source.
    Parameters:
        source (int or str): Index of the camera (e.g., 0 for default camera) or a video file path.
    Returns:
        cap (cv2.VideoCapture): OpenCV video capture object.
    """
  
    cap = cv2.VideoCapture(source)
    return cap


if __name__ == '__main__':
    shotdir = '.'  # shots are stored in the same directory as this script
    sources = [0]  # list of sources, here it's just the default camera

    # Corrected print statement for Python 3
    print('shotdir = {} \n'.format(shotdir))
    print("Keys: ESC = exit, SPACE = save frame to shot_0_000.bmp etc")

    # Create capture objects for all sources
    caps = list(map(create_capture, sources))  # use list() for Python 3 compatibility
    shot_idx = 0

    while True:
        imgs = []
        for i, cap in enumerate(caps):
            ret, img = cap.read()
            if ret:
                imgs.append(img)
                cv2.imshow('capture %d' % i, img)

        # Handle key presses
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:  # ESC key
            break
        if ch == ord(' '):  # SPACE key
            for i, img in enumerate(imgs):
                fn = '%s/shot_%d_%03d.bmp' % (shotdir, i, shot_idx)
                cv2.imwrite(fn, img)
                print(fn, 'saved')
            shot_idx += 1

    # Release resources and close windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
