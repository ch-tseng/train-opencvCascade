#python test-slidingwindow.py -c conf/office.json -i datasets/labelimages-1600/images/csi_1600x1600_20180124_130411.jpg

import time, os
import os.path
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Path for images for negative source")
ap.add_argument("-o", "--output", required=True, help="Path for output folder of negative images")
ap.add_argument("-w", "--width", required=True, type=int, help="Width for negative images")
ap.add_argument("-l", "--long", required=True, type=int, help="Height for negative images")
ap.add_argument("-t", "--type", required=True, help="Image type for source images, jpg or png")
ap.add_argument("-d", "--dtype", required=True, help="Image type for output negative images, jpg or png")

args = vars(ap.parse_args())

winW = args["width"]
winH = args["long"]

def imgPyramid(image, scale=0.5, minSize=[120,120], debug=False):
    yield image
 
        # keep looping over the pyramid
    while True:
        w = int(image.shape[1] * scale)
        h = int(image.shape[0] * scale)
        image = cv2.resize(image, (w, h))
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
 
        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

for file in os.listdir(args["source"]):
    filename, file_extension = os.path.splitext(file)

    print(file_extension.lower())
    if("." + args["type"] == file_extension.lower()):
        image = cv2.imread(args["source"]+"/"+file)

        # loop over the image pyramid
        for layer in imgPyramid(image, scale=0.5, minSize=[winW,winH]):
            print(layer.shape)
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in sliding_window(layer, stepSize=winW, windowSize=(winW, winH)):
                # if the current window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
 
                clone = layer.copy()
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                #cv2.imshow("Window", clone)

                cv2.imwrite(args["output"]+"/"+str(time.time())+"."+args["dtype"], window)
                #cv2.waitKey(1)
                #time.sleep(0.025)
