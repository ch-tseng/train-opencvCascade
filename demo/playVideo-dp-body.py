#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, os
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from scipy import io
import numpy as np
import imutils
import cv2
import sys
import time
import datetime
from keras.models import model_from_json
from skimage import io, transform

#-----------------------------------------------
cascadeFile = "body_LBA_48.xml"
useDP = True
writeIMG = False   #write all roi images
#sizeIMG = (128, 128)
maxPeopleInSeconds = 60  # secoonds, we will take the max number peoples as correct number in this time range.
peopleCount = []

videoPath = "mis-office-demo.mp4"
min_size = (90, 90)
max_size = (180, 180)
monitor_winSize = (640, 640)
savePath = "F:\\Video-misblock\\rois"
face_cascade = cv2.CascadeClassifier(cascadeFile)

#-----------------------------------------------------------

camera = cv2.VideoCapture(videoPath)
#camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])

def putText(image, text, x, y, color=(255,255,255), thickness=1, size=1.2):
    if x is not None and y is not None:
        cv2.putText( image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return image

def peopleNumerList(numPeople = 0):
    global peopleCount, maxPeopleInSeconds

    if(len(peopleCount)>maxPeopleInSeconds):
        peopleCount.pop(0)

    peopleCount.append(numPeople)


class dpModel:
    def __init__(self, dpmodel, dpweights):
        # load json and create model
        json_file = open(dpmodel, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(dpweights, by_name=True)
        print("Loaded model from disk")

    def reshaped_image(self, image):
        return transform.resize(image,(90, 90, 3)) # (cols (width), rows (height)) and don't u$

    def predict(self, img):
        test_image = []
        #img = cv2.imread("datasets/1/csi_1600x1600_20180614_083740.png-1.jpg")
        test_image.append(self.reshaped_image(img))
        test_image = np.array(test_image)
        print(test_image.shape)
        result = self.loaded_model.predict(test_image)
        print(result)
        return False if (result[0][1]>result[0][0]) else True

if(useDP==True):
    dp = dpModel("model.json", "model.h5")

while(camera.isOpened()):
    (grabbed, img) = camera.read()
    imgSource = img.copy()

    if(grabbed):	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor= 1.1,
            minNeighbors=6,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    

        i = 0
        predictROI = False
        r = monitor_winSize[1] / img.shape[1]
        dim = (monitor_winSize[0], int(img.shape[0] * r))
        img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		
        for (x,y,w,h) in faces:
	
            if( (w>min_size[0] and h>min_size[1]) and (w<max_size[0] and h<max_size[1]) ):
                print("(w, h) = ({}, {})".format(w,h))
                roi = imgSource[y:y+h, x:x+w]
                now=datetime.datetime.now()
                faceName = '%s_%s_%s_%s_%s_%s_%s.jpg' % (now.year, now.month, now.day, now.hour, now.minute, now.second, i)
                if(useDP == True):
                    predictROI = dp.predict(roi)
                    if(predictROI==True):
                        cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
						
                        i += 1
                        print("True")
                        putText(img2, "True", x+5, y+20, color=(255,0,0), thickness=2, size=0.6)
                        if(writeIMG==True):
                            cv2.imwrite(savePath+"\\1\\" + faceName, roi)
                        
						
                    else:
                        cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),1)
                        print("False")
                        putText(img2, "False", x+5, y+20, color=(0,0,255), thickness=2, size=0.6)
                        if(writeIMG==True):
                            cv2.imwrite(savePath+"\\0\\" + faceName, roi)
	
                    
                else:
                    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),3)
                    i += 1
                    if(writeIMG==True):
                        cv2.imwrite(savePath+"\\1\\" + faceName, roi)


        peopleNumerList(i) #Put the current people counts to array.
        #putText(img2, str(max(peopleCount))+" peoples", 25, 40, (0,0,255), thickness=2, size=1)
          

        cv2.imshow("Frame", img2)
        key = cv2.waitKey(1)
    else:
        print("loop...")
        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
