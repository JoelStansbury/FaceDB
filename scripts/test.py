import cv2
import os
import matplotlib.pyplot as plt

img_path = os.path.abspath('../data/images/')
imgs = []

# Gets the absolute path to each image.
# This saves us the hassle of appending the path to each img later on
for im in os.listdir(img_path):
    imgs.append(os.path.join(img_path,im))

# load the required trained XML classifiers
# https://github.com/opencv/opencv/tree/master/data/haarcascades/haarcascade_frontalface_default.xml
# Pretrained Face Detector
# (N*M) dimensional image --> list of rectangles (bounding boxes for each face detected)
face_cascade = cv2.CascadeClassifier('../data/models/haarcascade_frontalface_default.xml')

# # https://github.com/opencv/opencv/tree/master/data/haarcascades/haarcascade_eye.xml
# # Trained XML file for detecting eyes
# eye_cascade = cv2.CascadeClassifier('../data/models/haarcascade_eye.xml')

embedder = cv2.dnn.readNetFromTorch('../data/models/nn4.small2.v1.t7')

img = cv2.imread(imgs[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)


for (x,y,w,h) in faces:
    # To draw a rectangle in a face
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

cv2.imshow('img',img)
cv2.waitKey()
