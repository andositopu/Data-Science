#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import OpenCV2
import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Start video
vid_cam = cv2.VideoCapture(0)

# Detek objek
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# deteksi 1 wajah
face_id = 1

# Initialize sample
count = 0

assure_path_exists("dataset/")

while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert frame menjadi grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detek frames dan list rectangles
    faces = face_detector.detectMultiScale(gray, 1.1, minNeighbors= 5)

    for (x,y,w,h) in faces:

        # Crop the image frame
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # menambah sample face image
        count += 1

        # Save sample
        cv2.imwrite()

        # menampilkan sampel
        cv2.imshow('frame', image_frame)

    # melakukan looping untuk pengulangan
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif count>100:
        break

# Stop video
vid_cam.release()

#menutup windows
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




