import cv2
import math

counter = 0
cap = cv2.VideoCapture('path/to/file')   # filename. replace path by 0 for frame capture from webcam
fps = cap.get(5) # 5th propid returns frames per second from the video

while(cap.isOpened()):
    frame_id = cap.get(1) # 1st propid returns current frame index
    ret, frame = cap.read()
    if (frame_id % math.floor(fps) == 0):
        image ="frame_{}.jpg".format(counter)
        counter+=1
        cv2.imwrite(image, frame)
cap.release()
