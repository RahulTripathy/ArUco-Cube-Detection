import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture(0)

prev_frame_time = time.time()

image_count=0
frame_count=0

while True:

    ret, frame = cap.read()

    #Acquires frames periodically for calibration
    frame_count += 1
    if frame_count==30:
        cv.imwrite("Cal_Image"+str(image_count)+".jpg", frame)
        image_count = image_count+1
        frame_count = 0

    #Displays frame rate on the image as well as the frames
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    cv.imshow("Feed",frame)

    if cv.waitKey(1) & 0xFF == ord("q") : break

cap.release()
cv.destroyAllWindows()