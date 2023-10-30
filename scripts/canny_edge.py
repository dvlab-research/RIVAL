import cv2
import numpy as np

# Load the image
image = cv2.imread('assets/images/contorlnet_test/canny_ori_img.png', cv2.IMREAD_GRAYSCALE)

# Check if image loading is successful
if image is None:
    print('Could not open or find the image')
    exit(0)

# Apply Canny edge detection
edges = cv2.Canny(image, threshold1=150, threshold2=200)

# Save the result
cv2.imwrite('assets/images/contorlnet_test/canny_preprocess.png', edges)