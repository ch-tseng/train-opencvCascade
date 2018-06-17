import glob, os
import os.path
import time
#import argparse
import cv2
from xml.dom import minidom

xmlFolder = "datasets/head/1600x1600/labels"
imgFolder = "datasets/head/1600x1600/images"
labelName = "body"

#All files and setting will save to the project folder
folderPROJECT = "body1"

makeROI_dataset = True
roiSaveToFolder = "bodys"  #If empty, the roi image from the labels will not be saved.
roiSaveType = "jpg"
roiResize = (128,128)

#create to the label list
positiveFile = "positives.txt"

#create to background images (negative images) 
makeBackgroundImages = True
backgroundFolder = "bgs"
backgroundFile = "backgrounds.txt"

#make smaller negative images from the background images
makeNegativeDataset = True
negativeFolder = "negImages"  # all negative images will save to this folder
negativeSize = (128,128)
negativeType = "jpg"
maxImageUsed = 50  # How many negative images will be used (from background and neg source images)
negativeImageFolder = "negSource"  # source for more negative images, except the background images.
negativeFile = "negatives.txt"
#---------------------------------------------------------------------------

roiSaveToFolder = folderPROJECT + "/" + roiSaveToFolder
positiveFile = folderPROJECT + "/" + positiveFile
backgroundFolder = folderPROJECT + "/" + backgroundFolder
backgroundFile = folderPROJECT + "/" + backgroundFile
negativeFolder = folderPROJECT + "/" + negativeFolder
negativeFile = folderPROJECT + "/" + negativeFile

totalLabels = 0
wLabels = 0
hLabels = 0


def getLabels(imgFolder, xmlFilename, assignName=""):
    global totalLabels, wLabels, hLabels

    labelXML = minidom.parse(xmlFilename)
    labelName = []
    labelXstart = []
    labelYstart = []
    labelW = []
    labelH = []
    totalW = 0
    totalH = 0
    countLabels = 0

    tmpArrays = labelXML.getElementsByTagName("filename")
    for elem in tmpArrays:
        filenameImage = elem.firstChild.data
    print ("Image file: " + filenameImage)


    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        labelName.append(str(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmin")
    for elem in tmpArrays:
        labelXstart.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymin")
    for elem in tmpArrays:
        labelYstart.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmax")
    for elem in tmpArrays:
        labelW.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymax")
    for elem in tmpArrays:
        labelH.append(int(elem.firstChild.data))

    opencvCascade = ""
    tmpChars = ""
    num = 0

    image = cv2.imread(imgFolder + "/" + filenameImage)
    filepath = imgFolder
    filename = filenameImage

    if(makeROI_dataset==True):
        if not os.path.exists(roiSaveToFolder + "/"):
            os.makedirs(roiSaveToFolder + "/")

    print(len(labelName))
    for i in range(0, len(labelName)):
        if(assignName=="" or assignName==labelName[i]):
            #yield (labelName[i], labelXstart[i], labelYstart[i], int(labelW[i]-labelXstart[i]), int(labelH[i]-labelYstart[i])) 
            num += 1
            countLabels += 1
            totalW = totalW + int(labelW[i]-labelXstart[i])
            totalH = totalH + int(labelH[i]-labelYstart[i]) 
            tmpChars = tmpChars + "{} {} {} {}   ".format( labelXstart[i], labelYstart[i], int(labelW[i]-labelXstart[i]), int(labelH[i]-labelYstart[i]) )

            if(makeROI_dataset==True):
                roi = image[labelYstart[i]:labelH[i], labelXstart[i]:labelW[i]]
                resize = cv2.resize(roi, roiResize, interpolation=cv2.INTER_CUBIC) 
                cv2.imwrite(roiSaveToFolder + "/" + filename + "-" + str(i)+"."+roiSaveType, resize)

            cv2.rectangle(image, (labelXstart[i], labelYstart[i]), (labelXstart[i]+int(labelW[i]-labelXstart[i]), labelYstart[i]+int(labelH[i]-labelYstart[i])), (0,0,0), -1)
            #cv2.imshow("BG", image)
            #cv2.waitKey(0)
            #cv2.imshow("ROI", roi)
            #cv2.waitKey(0)

    if(countLabels>0):
        wLabels += totalW
        hLabels += totalH
        totalLabels += countLabels

        print("Average W, H: {}, {}".format(int(totalW/countLabels), int(totalH/countLabels)) )

        if(makeBackgroundImages==True):
            if not os.path.exists(backgroundFolder + "/"):
                os.makedirs(backgroundFolder + "/")

            cv2.imwrite(backgroundFolder+"/bg_" + filename, image)

        return "../{}  {}  {}".format(filepath+"/"+filename, num, tmpChars)

    else:
        return "0"

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


#--------------------------------------------------------------------------------
if not os.path.exists(folderPROJECT + "/"):
    os.makedirs(folderPROJECT + "/")

i = 0
with open(positiveFile, 'a') as the_file:

    for file in os.listdir(xmlFolder):
        filename, file_extension = os.path.splitext(file)

        if(file_extension==".xml"):
            print("XML: {}".format(filename))

            #imgfile = imgFolder+"/"+filename+"."+imageType
            xmlfile = xmlFolder + "/" + file
            print(xmlfile)
            #if(os.path.isfile(imgfile)):
            outLabels = getLabels(imgFolder, xmlfile, labelName)
            if(outLabels!="0"):
                the_file.write(outLabels + '\n')
                print( outLabels )
                print()

print("----> Average W:H = {}:{}".format(round(wLabels/totalLabels, 1), round(hLabels/totalLabels,1) ))

the_file.close()

with open(backgroundFile, 'a') as the_file:
    for file in os.listdir(backgroundFolder):
        filename, file_extension = os.path.splitext(file)

        if(file_extension.lower()==".jpg" or file_extension.lower()==".png" or file_extension.lower()==".jpeg"):
            the_file.write("./" + backgroundFolder + "/" + file + '\n')

the_file.close()

if(makeNegativeDataset==True):
    if not os.path.exists(negativeFolder + "/"):
        os.makedirs(negativeFolder + "/")

    usedImageCount = 0

    for folder in [backgroundFolder, negativeImageFolder]:

        for file in os.listdir(folder):
            if(usedImageCount>maxImageUsed):
                break

            filename, file_extension = os.path.splitext(file)

            if(file_extension.lower()==".jpg" or file_extension.lower()==".png" or file_extension.lower()==".jpeg"):
                image = cv2.imread(backgroundFolder + "/" + file)

                # loop over the image pyramid
                for layer in imgPyramid(image, scale=0.5, minSize=[negativeSize[0],negativeSize[1]]):
                    print(layer.shape)
                    # loop over the sliding window for each layer of the pyramid
                    for (x, y, window) in sliding_window(layer, stepSize=negativeSize[0], windowSize=negativeSize):
                        # if the current window does not meet our desired window size, ignore it
                        if window.shape[0] != negativeSize[1] or window.shape[1] != negativeSize[0]:
                            continue
 
                        clone = layer.copy()
                        cv2.rectangle(clone, (x, y), (x + negativeSize[0], y + negativeSize[1]), (0, 255, 0), 2)

                        cv2.imwrite(negativeFolder + "/" + str(time.time()) + "." + negativeType, window)

            usedImageCount += 1 

with open(negativeFile, 'a') as the_file:
    for file in os.listdir(negativeFolder):
        filename, file_extension = os.path.splitext(file)

        if(file_extension.lower()==".jpg" or file_extension.lower()==".png" or file_extension.lower()==".jpeg"):
            the_file.write("./" + backgroundFolder + "/" + file + '\n')

the_file.close()

if not os.path.exists(folderPROJECT + "/cascad_data/"):
    os.makedirs(folderPROJECT + "/cascad_data/")
