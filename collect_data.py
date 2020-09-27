# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:46:05 2020

@author: LENOVO
"""

import cv2
import urllib
import numpy as np

face_data = r"haarcascade_frontalface_default.xml"

classifier = cv2.CascadeClassifier(face_data)

url = "http://192.168.43.1:8080/shot.jpg"

data = []

while len(data)<100:
    image_from_url = urllib.request.urlopen(url)
    frame = np.array(bytearray(image_from_url.read()),np.uint8)
    frame = cv2.imdecode(frame,-1)
    face_frame = frame.copy()
    
    faces = classifier.detectMultiScale(frame,1.3,5)
    
    if len(faces)>0:
        for x,y,w,h in faces:
            face_frame = frame[y:y+h,x:x+w].copy()
            cv2.imshow("only_face",face_frame)
            if len(data)<100:
                print(len(data)+1,"/100")
                data.append(face_frame)
            else:
                break
    cv2.imshow("face",frame)
    if cv2.waitKey(30)== ord("a"):
        break
    
    
cv2.destroyAllWindows()

if len(data) == 100:
    name = input("enter user name : ")
    for i in range(100):
        cv2.imwrite("images/"+name+"_"+str(i)+".jpg",data[i])
    print("Done")

else:
    print("need more data")