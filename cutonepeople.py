# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 19:39:56 2020

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 18:24:24 2020

@author: ASUS
"""

import numpy as np
import cv2

# Open the video
cap = cv2.VideoCapture('D:/vellas/MatchNMingle/full videos/day1/cam6_part1/table1/cam6_table1_couple3.mp4')

# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Here you can define your croping values

x,y,h,w = 360,50,400,380

# output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D:/vellas/MatchNMingle/full videos/day1/cam6_part1/table1/split/.mp4', fourcc, fps, (w, h))


# Now we start
while(cap.isOpened()):
    ret, frame = cap.read()

    cnt += 1 # Counting frames

    # Avoid problems when video finish
    if ret==True:
        # Croping the frame
        crop_frame = frame[y:y+h, x:x+w]
        

        # Percentage
        xx = cnt *100/frames
        print(int(xx),'%')

        # Saving from the desired frames
        #if 15 <= cnt <= 90:
        #    out.write(crop_frame)

        # I see the answer now. Here you save all the video
        out.write(crop_frame)

        # Just to see the video in real time          
        cv2.imshow('frame',frame)
        cv2.imshow('croped',crop_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()