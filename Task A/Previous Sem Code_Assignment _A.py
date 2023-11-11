# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 00:05:30 2022

@author: Jay
"""

import cv2
import numpy as np

#Importing Video
vs = cv2.VideoCapture("street.mp4")
of = cv2.VideoCapture("office.mp4")
ex = cv2.VideoCapture("exercise.mp4")
yt = cv2.VideoCapture("talking.mp4")

#Getting total frames number
total_no_frames_VS = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
total_no_frames_VO = int(of.get(cv2.CAP_PROP_FRAME_COUNT))
total_no_frames_VE = int(ex.get(cv2.CAP_PROP_FRAME_COUNT))

# total_no_frames_yt 
yt_frame = int(yt.get(cv2.CAP_PROP_FRAME_COUNT))

#Video Output Format Define
VS = cv2.VideoWriter('yt_on_street.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),30.0,(1280,720))
VO = cv2.VideoWriter('yt_on_office.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),30.0,(1280,720))
VE = cv2.VideoWriter('yt_on_exercise.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),30.0,(1280,720))
# yt_out = cv2.VideoWriter('yt.avi',
#                       cv2.VideoWriter_fourcc(*'MJPG'),30.0,(640, 360))


for frame_count in range(total_no_frames_VS):
    success, frame = vs.read()
    [nrow2, ncol2, nlayer2] = frame.shape
    output_frame = np.ones((nrow2, ncol2, nlayer2), dtype = np.float64)
    success_2, yt_frame = yt.read()
    output_frame *= frame
    
    face_cascade = cv2.CascadeClassifier("face_detector.xml")
    faces = face_cascade.detectMultiScale(frame, 1.21, 6)
                
    for (x, y, w, h) in faces:
        output_frame[y : y + h, x : x + w] = cv2.blur(frame[y : y + h, x : x + w], (25, 25), cv2.BORDER_REPLICATE)
        output_frame = output_frame.astype(np.uint8)
        
    if frame_count < 465:
        [nrow, ncol, nlayer] = yt_frame.shape
        yt_frame_copy = cv2.resize(yt_frame, [426, 240])
        yt_frame_copy2 = cv2.copyMakeBorder(yt_frame_copy, 15, 15,15, 15, cv2.BORDER_CONSTANT, None, value = 0)
        [nrow, ncol, nlayer] = yt_frame_copy2.shape
        output_frame[100: 100+270, 100: 100+456] = yt_frame_copy2
        
    VS.write(output_frame)