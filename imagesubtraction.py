# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 02:55:21 2020

@author: ASUS
"""

# Python programe to illustrate  
# arithmetic operation of 
# subtraction of pixels of two images 
  
# organizing imports  
import cv2  
import numpy as np  
    
# path to input images are specified and   
# images are loaded with imread command  
image1 = cv2.imread('D:/YZU/dataset/MatchnMingle/speed_dates/fix code/data/frame0.jpg')
image2 = cv2.imread('D:/YZU/dataset/MatchnMingle/speed_dates/fix code/data/frame1.jpg')
  
# cv2.subtract is applied over the 
# image inputs with applied parameters 
sub = cv2.subtract(image1, image2) 

Conv_hsv_Gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
sub[mask != 255] = [0, 0, 255]
image1[mask != 255] = [0, 0, 255]
image2[mask != 255] = [0, 0, 255]
# the window showing output image 
# with the subtracted image  
cv2.imshow('Subtracted Image', sub) 
print(image1)
print(image2)
print(sub)
  
# De-allocate any associated memory usage   
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()




