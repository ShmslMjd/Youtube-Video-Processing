import cv2
import numpy as np
from matplotlib import pyplot as pt

# Load the image
image = cv2.imread('001.png', 0)

# Threshold the image
_, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

# Apply dilation
kernel = np.ones((15,15), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate dynamic threshold
threshold = np.median([cv2.contourArea(cnt) for cnt in contours])

# Filter contours based on area
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold]

# Get bounding rectangles for each contour
rects = [cv2.boundingRect(cnt) for cnt in contours]

# Sort rectangles top-to-bottom, left-to-right
rects.sort(key=lambda x: (x[1], x[0]))

# Extract and display each paragraph
for i, rect in enumerate(rects):
    x, y, w, h = rect
    paragraph = image[y:y+h, x:x+w]
    pt.figure(figsize=(10,10))
    pt.imshow(paragraph, cmap='gray')
    pt.axis('off')
    pt.show()
