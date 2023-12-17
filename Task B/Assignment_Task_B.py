import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('001.png', 0)

# Threshold the image
_, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

# Apply dilation
kernel = np.ones((15, 15), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=2)

# Find contours
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

