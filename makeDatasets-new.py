import glob, os
import os.path
import time
import argparse
import cv2
from xml.dom import minidom
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#==============================================================
xmlFolder = "datasets/head/1600x1600/labels"
imgFolder = "datasets/head/1600x1600/images"
labelName = "head"

numAUG = 3
numNegAUG = 5

maxMaskedBackgrounds = 5000
maxNegSource = 60

folderPROJECT = "head"

imageKeepType = "jpg"
positiveDesc = "positives.info"
positiveDesc_aug = "positives-aug.info"
positiveFolder = "pos-images"
negativeDesc = "negatives.info"
negativeDesc_aug = "negatives-aug.info"
negativeFolder = "neg-images"

negSource = "datasets/negSource"
vecImage = folderPROJECT + ".vec"
maskedBackgrounds = "masked"

#==============================================================

positiveDesc = folderPROJECT + "/" + positiveDesc
negativeDesc = folderPROJECT + "/" + negativeDesc
positiveDesc_aug = folderPROJECT + "/" + positiveDesc_aug
negativeDesc_aug = folderPROJECT + "/" + negativeDesc_aug

positiveFolder = folderPROJECT + "/" + positiveFolder
negativeFolder = folderPROJECT + "/" + negativeFolder
maskedBackgrounds = folderPROJECT + "/" + maskedBackgrounds

totalLabels = 0
wLabels = 0
hLabels = 0

#Create all required folders
if not os.path.exists(positiveFolder + "/"):
    os.makedirs(positiveFolder + "/")

if not os.path.exists(negativeFolder + "/"):
    os.makedirs(negativeFolder + "/")

if not os.path.exists(maskedBackgrounds + "/"):
    os.makedirs(maskedBackgrounds + "/")

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


def createPositive(imgFolder, xmlFilename, descFile, assignName=""):
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
    countLabels = 0

    image = cv2.imread(imgFolder + "/" + filenameImage)
    image2 = image.copy()
    filepath = imgFolder
    filename = filenameImage

#    with open(descFile, 'a') as the_file_aug:
    for i in range(0, len(labelName)):
        if(assignName=="" or assignName==labelName[i]):
            countLabels += 1
            totalW = totalW + int(labelW[i]-labelXstart[i])
            totalH = totalH + int(labelH[i]-labelYstart[i]) 
            rois = "{} {} {} {}   ".format( labelXstart[i], labelYstart[i], int(labelW[i]-labelXstart[i]), int(labelH[i]-labelYstart[i]) )
            rois_aug = "{} {} {} {}   ".format( 0, 0, int(labelW[i]-labelXstart[i]), int(labelH[i]-labelYstart[i]) )
            tmpChars = tmpChars + rois

            #get the label image from the source image
            roi = image[labelYstart[i]:labelH[i], labelXstart[i]:labelW[i]]
            roiFile = positiveFolder + "/" + str(totalLabels) + "-" + str(countLabels)+"."+imageKeepType
            roi_augFile = positiveFolder + "/aug_" + str(totalLabels)
            cv2.imwrite(roiFile, roi)
            #the_file_aug.write("{}  {}  {}\n".format(roiFile, 1, rois_aug) )

            #negative/background files, use mask
            cv2.rectangle(image2, (labelXstart[i], labelYstart[i]), (labelXstart[i]+int(labelW[i]-labelXstart[i]), labelYstart[i]+int(labelH[i]-labelYstart[i])), (0,0,0), -1)

            #Image augmentation
            datagen = ImageDataGenerator(
                zca_whitening=False,
                rotation_range=360,
                #width_shift_range=0.2,
                #height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode="nearest")

            roi = roi[...,::-1]
            x = img_to_array(roi)   # this is a Numpy array with shape (3, 150, 150) 
            x = x.reshape (( 1 ,) + x.shape)   # this is a Numpy array with shape (1, 3, 150, 150)

            i =  0 
            for batch in datagen.flow(x, batch_size = 1 ,
                save_to_dir = positiveFolder, save_prefix = 'aug', save_format = 'jpg' ):
                #the_file_aug.write(roi_augFile + "_" + str(i) + "." + imageKeepType + "   1  0 0 " + str(x.shape[1]) + " " + str(x.shape[2]) + '\n')
                i +=  1 
                if i >  numAUG :
                    break   # otherwise the generator would loop indefinitely

    #the_file_aug.close()
    cv2.imwrite(maskedBackgrounds+'/'+filename, image2)

    wLabels += totalW
    hLabels += totalH
    totalLabels += countLabels

    print("Average W, H: {}, {}".format(int(totalW/countLabels), int(totalH/countLabels)) )

    return "../{}  {}  {}".format(filepath+"/"+filename, countLabels, tmpChars)


with open(positiveDesc, 'a') as the_file:

    #make positive images
    for file in os.listdir(xmlFolder):
        filename, file_extension = os.path.splitext(file)

        if(file_extension==".xml"):
            print("XML: {}".format(filename))

            #imgfile = imgFolder+"/"+filename+"."+imageType
            xmlfile = xmlFolder + "/" + file
            print(xmlfile)
            #if(os.path.isfile(imgfile)):
            outLabels = createPositive(imgFolder, xmlfile, positiveDesc_aug, labelName)
            the_file.write(outLabels + '\n')
            print( outLabels )
            print()

    avgW = round(wLabels/totalLabels, 1)
    avgH = round(hLabels/totalLabels,1)
    print("----> Average W:H = {}:{}".format(avgW, avgH ))
    negativeSize = (int(avgW * 1.2), int(avgH * 1.2) )

    #make negative images
    usedImageCount = 0
    count = 0
    foldercount = 0

    for folder in [maskedBackgrounds, negSource]:

        for file in os.listdir(folder):
            if(foldercount==0):
                maxNum = maxMaskedBackgrounds
            else:
                maxNum = maxNegSource

            foldercount += 1

            if(usedImageCount>maxNum):
                break

            filename, file_extension = os.path.splitext(file)

            if(file_extension.lower()==".jpg" or file_extension.lower()==".png" or file_extension.lower()==".jpeg"):
                image = cv2.imread(maskedBackgrounds + "/" + file)

                # loop over the image pyramid
                for layer in imgPyramid(image, scale=0.2, minSize=[negativeSize[0],negativeSize[1]]):
                    #print(layer.shape)
                    # loop over the sliding window for each layer of the pyramid
                    for (x, y, window) in sliding_window(layer, stepSize=negativeSize[0], windowSize=negativeSize):
                        # if the current window does not meet our desired window size, ignore it
                        if window.shape[0] != negativeSize[1] or window.shape[1] != negativeSize[0]:
                            continue
 
                        #clone = layer.copy()
                        #cv2.rectangle(clone, (x, y), (x + negativeSize[0], y + negativeSize[1]), (0, 255, 0), 2)
                        #Image augmentation
                        datagen = ImageDataGenerator(
                            zca_whitening=False,
                            rotation_range=360,
                            #width_shift_range=0.2,
                            #height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode="nearest")

                        #roi = roi[...,::-1]
                        x = img_to_array(window)   # this is a Numpy array with shape (3, 150, 150) 
                        x = x.reshape (( 1 ,) + x.shape)   # this is a Numpy array with shape (1, 3, 150, 150)

                        i =  0 
                        for batch in datagen.flow(x, batch_size = 1 ,
                            save_to_dir = negativeFolder, save_prefix = 'aug', save_format = imageKeepType ):
                            #the_file_aug.write(roi_augFile + "_" + str(i) + "." + imageKeepType + "   1  0 0 " + str(x.shape[1]) + " " + str(x.shape[2]) + '\n')
                            i +=  1 
                            count += 1
                            if i >  numNegAUG :
                                break   # otherwise the generator would loop indefinitely



                        #cv2.imwrite(negativeFolder + "/" + str(count) + "_" + str(time.time()) + "." + imageKeepType, window)

            usedImageCount += 1 

with open(negativeDesc, 'a') as the_file:
    for file in os.listdir(negativeFolder):
        filename, file_extension = os.path.splitext(file)

        if(file_extension.lower()==".jpg" or file_extension.lower()==".png" or file_extension.lower()==".jpeg"):
            the_file.write(negativeFolder + "/" + file + '\n')

the_file.close()


with open(positiveDesc_aug, 'a') as the_file:
    for file in os.listdir(positiveFolder):
        filename, file_extension = os.path.splitext(file)
        if(file_extension.lower()==".jpg" or file_extension.lower()==".png" or file_extension.lower()==".jpeg"):
            print(positiveFolder + "/" + file)
            img = cv2.imread(positiveFolder + "/" + file)
            sizeimg = img.shape
            the_file.write( "../" + positiveFolder + "/" + file + '  1  0 0 ' + str(sizeimg[1]) + ' ' + str(sizeimg[0]) + '\n')

the_file.close()



print("----> Average W:H = {}:{}".format(round(wLabels/totalLabels, 1), round(hLabels/totalLabels,1) ))

the_file.close()

if not os.path.exists(folderPROJECT + "/cascad_data/"):
    os.makedirs(folderPROJECT + "/cascad_data/")

