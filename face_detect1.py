#!/usr/bin/env python
# coding: utf-8

# In[2]:


import face_recognition
import os
import cv2
import numpy as np


# In[ ]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
vedio=cv2.VideoCapture(0)
file_name=input()
if os.path.isdir("known"):
	os.mkdir(f"known/{file_name}")
else:
	os.mkdir("known")

i=0
while True:
    ret,img=vedio.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        image=img[y:y+h,x:x+w]
        i+=1
        if i<100:
        	cv2.imwrite(f"known/{file_name}/{i}.jpg",image)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    key=cv2.waitKey(1)
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()


# In[ ]:




