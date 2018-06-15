import numpy as np
import cv2
import glob
import os

sourePath = "photos.1600"
savePath = "faces"
face_size_min = (24, 24)

if not os.path.exists(savePath):
    os.makedirs(savePath)
	
face_cascade = cv2.CascadeClassifier('cascad_data/cascade.xml')

i = 0

for filePath in glob.glob(sourePath+"/*.jpg"):
    print("Load {} ...".format(filePath))
    
    img = cv2.imread(filePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 8)

    for (x,y,w,h) in faces:
        if(w>=face_size_min[0] and h>=face_size_min[1]):
            roi_color = img[y:y+h, x:x+w]
            cv2.imwrite(savePath+"/face-"+str(i)+".jpg", roi_color)
            i+=1
			
print("All faces has been extracted to {}".format(savePath))
