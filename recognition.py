#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import os
import cv2
import numpy as np


# In[2]:


known="known"
tolerance=0.3
frame=3
font=2
model="large"
video=cv2.VideoCapture(0)
print("loading faces")
known_faces=[]
known_names=[]
for name in os.listdir(known):
    for filename in os.listdir(f"{known}/{name}"):
        image=face_recognition.load_image_file(f"{known}/{name}/{filename}")
        encoding=face_recognition.face_encodings(image)
        if not len(encoding):
            print(filename, "can't be encoded")
            continue
        known_faces.append(encoding[0])
        known_names.append(name)
i=0
while True:
    ret,img=video.read()
    locations=face_recognition.face_locations(img,model=model)
    encodings=face_recognition.face_encodings(img,locations)
    for face_encoding , face_location in zip (encodings,locations):
        results=face_recognition.compare_faces(known_faces,face_encoding,tolerance)
        match=None
        if True in results:
            match=known_names[results.index(True)]
            print(f"match found:::{match}")
            top_left=(face_location[3],face_location[0])
            bottom_right=(face_location[1],face_location[2])
            color=[0,255,0]
            cv2.rectangle(img,top_left,bottom_right,color,frame)
            top_left=(face_location[3],face_location[0])
            bottom_right=(face_location[1],face_location[2])
            color=[0,255,0]
            cv2.rectangle(img,top_left,bottom_right,cv2.FILLED)
            cv2.putText(img,match,(face_location[3]*10,face_location[2]*15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,0))
            i+=1
            
            cv2.imwrite(f"known/{match}/new{i}.jpg",img)
            #image=face_recognition.load_image_file(f"{known}/{name}/{filename}")
            encoding=face_recognition.face_encodings(img)
            if not len(encoding):
                print(filename, "can't be encoded")
                continue
            known_faces.append(encoding[0])
            known_names.append(match)
    cv2.imshow("recognition",img)    
    cv2.waitKey(2)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
                        
            


# In[ ]:




