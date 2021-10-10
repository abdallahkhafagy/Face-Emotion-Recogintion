# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 14:17:37 2021

@author: Admin
"""
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle


img = imread('data/face0_rotate.jpg')
classifier = CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces = classifier.detectMultiScale(img)

for face in faces:
    x, y, width, height = face
    print(x,width)
    x2, y2 = x + width, y + height
    
    rectangle(img, (x,y), (x2, y2), (0,0,255),2)
    
imshow('face detection', img)

waitKey(0)

destroyAllWindows()

