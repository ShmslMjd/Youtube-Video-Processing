# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:55:02 2022

@author: ysher
"""

import cv2
import numpy as np
from matplotlib import pyplot as pt

vid = cv2.VideoCapture("street.mp4")
out = cv2.VideoWriter('processed_video.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30.0,
                      (1280, 720))


# watermark_1 = cv2.imread("watermark1.png", 1)
# watermark_2 = cv2.imread("watermark2.png", 1)

# [nrow, ncol, nlayer] = watermark_1.shape
# bi_img = np.ones((nrow, ncol, nlayer), dtype=np.uint8)

# watermark_1_grayscale = cv2.cvtColor(watermark_1, cv2.COLOR_BGR2GRAY)
# watermark_1_hist = cv2.calcHist([watermark_1_grayscale], [0], None, [256], [0, 256])

# threshold = 36


face_cascade = cv2.CascadeClassifier("face_detector.xml")


total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_count in range(total_no_frames):
    print(frame_count)
    success, frame = vid.read()
    [nrow, ncol, nlayer] = frame.shape
    output_frame = np.ones((nrow, ncol, nlayer), dtype = np.uint8)
    output_frame *= frame
    faces = face_cascade.detectMultiScale(frame, 1.21, 6)
    for (x, y, w, h) in faces:
        output_frame[y : y + h, x : x + w] = cv2.blur(frame[y : y + h, x : x + w], (11, 11), cv2.BORDER_REPLICATE)
    
        

    for x in range(nrow):
        for y in range(ncol):
            if watermark_1_grayscale[x, y] > threshold:
                output_frame[x, y] = watermark_1[x, y]
                
    out.write(output_frame)
            
