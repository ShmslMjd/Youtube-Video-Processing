import cv2
import numpy as np
from matplotlib import pyplot as plt

# This is used to load the targeted image
image = cv2.imread('001.png', 0)

# Thresholding was used here to reduce any pixels below the range of 200 and anything above will be increased.
_, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

# Using dilation - This is used to increase the area of the object to make it bigger.
kernel = np.ones((15, 15), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=2)

# This is used to find contours in image basically meaning finding colours of same intensity.
# Later the dilated function is used for when the contours are found. 
# cv2.RETR_EXTERNAL is used to retrieve contour.
# cv2.CHAIN_APPROX_SIMPLE is used to compresses horizontal, vertical, and diagonal segments and leaves only their end points.
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate dynamic threshold
areas = [cv2.contourArea(cnt) for cnt in contours]
mean_area = np.mean(areas)
std_area = np.std(areas)
dynamic_threshold = mean_area - 2.0 * std_area  # Adjust the factor as needed

# Filter contours based on area
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > dynamic_threshold]

# Get bounding rectangles for each contour
rects = [cv2.boundingRect(cnt) for cnt in contours]

# Sort rectangles top-to-bottom, then left-to-right
rects.sort(key=lambda x: (x[0], x[1]))

# Extract and display each paragraph
for i, rect in enumerate(rects):
    x, y, w, h = rect
    paragraph = image[y:y+h, x:x+w]
    plt.figure(figsize=(10, 10))
    plt.imshow(paragraph, cmap='gray')
    plt.axis('off')
    plt.show()

