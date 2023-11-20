# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:58:15 2022

@author: ysher
"""



import cv2
import numpy as np
from matplotlib import pyplot as pt



"""
List of variables being declared

interval: refers to the number of frames needed before watermarks are switched
watermarks: list containing different watermarks to be used
watermark_selector: points to current watermark being used


CODE BELOW MIGHT PROMPT ERRORS IF RUN IN SYPDER VERSION 5.1.5, ONLY WORKS IF SPYDER VERSION IS 5.3.3. IF NOT UPDATED
COMMENT OUT THE CODE BELOW

IF SPYDER VERSION IS 5.1.5, MANUAL ENTRY OF VIDEOS IS REQUIRED IN ORDER TO WORK. DECOMMENT OUT vid VARIABLE AND ENTER
DESIRED VIDEO TO BE PROCESSED
"""



# selected_vid = input("\nPlease select the video you'd like to modify ('street.mp4', 'exercise.mp4', 'office.mp4') ")
# vid = cv2.VideoCapture(selected_vid)

# COMMENT OUT ABOVE CODE, IF SPYDER VERSION IS LESS THAN 5.3.3



vid = cv2.VideoCapture("office.mp4")
talking = cv2.VideoCapture("talking.mp4")
end = cv2.VideoCapture("endscreen.mp4")

out = cv2.VideoWriter('processed_video2.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30.0,
                      (1280, 720))

face_cascade = cv2.CascadeClassifier("face_detector.xml")

total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
total_overlay_frames = int(talking.get(cv2.CAP_PROP_FRAME_COUNT))

interval = total_no_frames//6

watermark1 = cv2.imread("watermark1.png", 1)
watermark2 = cv2.imread("watermark1.png", 1)



"""
The code below is simply used to obtain a histogram to check for a suitable threshold value
"""

# watermark1_histrogram = cv2.calcHist([watermark1], [0], None, [256], [0, 256])

# pt.figure()
# pt.title("Watermark 1's histogram")
# pt.xlabel("Bins")
# pt.ylabel("Number of Pixel")
# pt.xlim([0,256])
# pt.plot(watermark1_histrogram)



watermarks = [watermark1, watermark2]
watermark_selector = 1

print("\nVideo processing has began!\n")



""" 
Main processing occers below


The function loops through the total number of frames present in the video to be edited. Program detects if
watermark has to be switched with the use of frame_count and interval variables. A copy of video frame is made and
faces in the frame copy is blurred and resized in order to accomade watermarks. Finally the overlay video is placed.

"""



for frame_count in range(total_no_frames):
    
    # if (frame_count % interval == 0):
    print("Completion status: {}%".format(round(frame_count/total_no_frames * 100)), frame_count)
        # watermark_selector = (watermark_selector + 1) % 2
        
    success, frame = vid.read()

    
    if (frame_count <= total_overlay_frames):
        overlay_success, overlay_frame = talking.read()
    
    [nrow, ncol, nlayer] = frame.shape
    
    frame_copy = np.ones((nrow, ncol, nlayer), dtype=np.uint8)
    frame_copy *= frame
    faces = face_cascade.detectMultiScale(frame_copy, 1.129, 6)
   
    for (x, y, w, h) in faces:
        frame_copy[y : y + h, x : x + w] = cv2.blur(frame_copy[y : y + h, x : x + w], (15, 15), cv2.BORDER_REPLICATE)
        
    frame_copy = cv2.resize(frame_copy, (1280, 720))
    
    [nrow, ncol, nlayer] = frame_copy.shape
    
    for x in range(nrow):
        for y in range(ncol):
            if watermarks[watermark_selector][x, y][0] > 0:
                frame_copy[x, y] = watermarks[watermark_selector][x, y] 
    
    
    if (overlay_success):
        overlay_frame = cv2.resize(overlay_frame, [356, 200])
        overlay_frame = cv2.copyMakeBorder(overlay_frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, value = 0)
        frame_copy[50 : 50 + 210, 50 : 50 + 366] = overlay_frame
     

    
    out.write(frame_copy)
out.release()

print("\nVideo has been processed!")