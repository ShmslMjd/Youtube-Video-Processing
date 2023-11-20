import cv2
import numpy as np
from matplotlib import pyplot as pt

#Importing Video
youtuber = cv2.VideoCapture("talking.mp4")
traffic = cv2.VideoCapture("traffic.mp4")
crowd = cv2.VideoCapture("singapore.mp4")
grandma = cv2.VideoCapture("alley.mp4")
office = cv2.VideoCapture("office.mp4")
end = cv2.VideoCapture("endscreen.mp4")

#Getting total frames number
total_no_frames_youtuber = int(youtuber.get(cv2.CAP_PROP_FRAME_COUNT))
total_no_frames_traffic = int(traffic.get(cv2.CAP_PROP_FRAME_COUNT))
total_no_frames_crowd = int(crowd.get(cv2.CAP_PROP_FRAME_COUNT))
total_no_frames_grandma = int(grandma.get(cv2.CAP_PROP_FRAME_COUNT))
total_no_frames_office = int(office.get(cv2.CAP_PROP_FRAME_COUNT))
total_no_frames_end = int(end.get(cv2.CAP_PROP_FRAME_COUNT))