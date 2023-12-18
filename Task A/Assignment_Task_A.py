import cv2
import numpy as np
from matplotlib import pyplot as pt

#========================== Function Declaration ==========================#

# Check the average brightness of the video, the function loop through all the frames in the
# video and store in the frames [] list. The average brightness is calculated by taking the
# mean of the frame list.

def check_brightness(video):
    frames = []
    vid_bright = cv2.VideoCapture(video)
    total_no_frames = int(vid_bright.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_count in range(total_no_frames):
        ret, frame = vid_bright.read()
        frames.append(frame)
    vid_bright.release()
    
    avg_brightness = np.mean(frames)
    threshold = 100
    
    if avg_brightness > threshold:
        return "day"
    else:
        return "night"
    
# Increase the brightness of a frame, the function will received a frame and the frame
# will be converted from RGB to HSV as is a much easier to modify brightness in HSV color
# space because it has one channel that control the brightness. Modified frame will be
# converted back to RGB and will be returned.

def increase_brightness(frame, factor):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    v = hsv[:, :, 2]
    v = np.clip(v * factor, 0, 255)
    hsv[:, :, 2] = v
    brightened_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return brightened_frame

#========================== Main Processing ==========================#

# The videos will be stored in array and for loop will be used to process each video.
# At the start of the loop check_brightness() is called to determine the time of the
# video. Total number of frames is calculated and each frame will be processed and edited.
# During processing the program will check the value of brightness, if the value is 'night'
# increase the brightness of the whole video. The processed frame then will be passed to
# detectMultiScale() function to apply blur effect on the faces in the video.
# Next, the two watermarks image will be stored in array and the processed frame will be 
# overlayed by watermark. There will be 2 different 
# watermarks but each processed video will only have 1 watermark.
# It will be followed by the youtuber video being overlayed on top left of the video and 
# have black borders around it. Finally, every processed video will be added with "endscreen.mp4".
# After all processes done, all processed video will be produced.

# Videos stored in array
videos = ["singapore.mp4", "traffic.mp4", "alley.mp4", "office.mp4"]

# Watermarks stored in array
watermark1 = cv2.imread("watermark1.png", 1)
watermark2 = cv2.imread("watermark2.png", 1)
watermarks = [watermark1, watermark2]
watermark_selector = 1

# Counter for "videos" array
counter = 1

#Display in console to update the user
print("\nVideo processing has began!\n")

#For-loop to choose 1 video from the array
for video in videos:
    
    # Perform Brightness Check on the selected video
    brightness = check_brightness(video)
    
    # Capture the chosen video
    vid = cv2.VideoCapture(video)
    # Capture the overlaying video
    youtuber = cv2.VideoCapture("talking.mp4")
    # Count the frames in the overlaying video
    total_no_frames_youtuber = int(youtuber.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    vid_name = video.split('.')[0]
    #
    processed_video_name = f"processed_video_{vid_name}"
    # Create a variable, use Video Writer to set the output video name, format, frame count & spatial resolution
    result = cv2.VideoWriter(f"processed_video_{vid_name}.avi",
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          30.0,
                          (1280, 720))
    
    # Choose the file for face detection
    face_cascade = cv2.CascadeClassifier("face_detector.xml")

    # Count the frames of the chosen video
    total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # For-Loop to perform process for every frame of the chosen video
    for frame_count in range(total_no_frames):
        
        # Display in console to update the progress to the user
        print("Video {}".format(counter) + " completion status: {}%".format(round(frame_count/total_no_frames * 100)), frame_count)

        # Set the video frame into the variable "frame" if success to read the chosen video
        success, frame = vid.read()

        # If-else loop to determine which set of frames to use
        if brightness == "night":
            frame_result = increase_brightness(frame, 1.8)
        else:
            frame_result = frame

        # Detect multiple face in the processed frames
        faces = face_cascade.detectMultiScale(frame_result, 1.3, 6)
        
        # For-loop to apply blur to all detected faces
        for (x, y, w, h) in faces:
            frame_result[y : y + h, x : x + w] = cv2.blur(frame_result[y : y + h, x : x + w], (15, 15), cv2.BORDER_REPLICATE)

        # Read the spatial resolution and plane numbers of the processed frames
        [nrow, ncol, nlayer] = frame_result.shape
        
        # Divide the total number of frames of the chosen video from the list with 6
        # This would generate (in form of number of frames) the length of one section out of the created 6
        # This would generate unique interval time for each video
        # Alternatively, "interval" can be set to a static number (which indicates the number of frame), example: 'interval = 60' 
        interval = total_no_frames/6
        
        # For-loop to apply watermark to the processed video
        for x in range(nrow):
            for y in range(ncol):
                if (frame_count <= interval) or (frame_count > 2*interval and frame_count <= 3*interval) or (frame_count > 4*interval and frame_count <= 5*interval):
                    if watermarks[0][x, y][0] > 0:
                        frame_result[x, y] = watermarks[0][x, y]
                else:
                    if watermarks[1][x, y][0] > 0:
                        frame_result[x, y] = watermarks[1][x, y]

        # If statement to read overlaying video when the frame count of the chosen video is lesser than the total number of frames of the overlaying video
        if (frame_count <= total_no_frames_youtuber):
            overlay_success, overlay_frame = youtuber.read()

        # If statement to resize every frame of overlaying video, add black border around the frames, 
        # position the overlaying frame if reading the overlay video was success
        if (overlay_success):
            overlay_frame = cv2.resize(overlay_frame, [356, 200])
            overlay_frame = cv2.copyMakeBorder(overlay_frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, value = 0)
            frame_result[50 : 50 + 210, 50 : 50 + 366] = overlay_frame

        # Write the processed frames into the created Video Writer variable
        result.write(frame_result)

    # Capture the endscreen video
    endscreen = cv2.VideoCapture("endscreen.mp4")

    # Calculate the total number of frames in endscreen video 
    total_no_frames_endscreen = int(endscreen.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # For-loop to write the frames of endscreen video into the created Video Writer variable
    for frame_count in range(total_no_frames_endscreen):
        success, frame = endscreen.read()
        result.write(frame)

    # Display in console to update the user
    print(f"Video {counter} has been processed")
    
    # Increase the counter number to choose the next video in the list
    counter += 1
    
    # The frames of overlay video has to be release to avoid confusion in overlaying on the next video.
    youtuber.release()

    # Generate the video file in the same directory
    result.release()
    
print("\nAll video has been processed!")
