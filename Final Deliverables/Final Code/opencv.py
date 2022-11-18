import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
import trainlist

cap=cv.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset=20
img_size=300
classifier=Classifier("./Model/keras_model.h5","./Model/labels.txt")
labels=trainlist.dataset

rec=0


def get_frame():
    count=0
    index_l=-1
    ret,img=cap.read()
    img_out=img.copy()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        #Image empty
        img_bg=np.ones((img_size,img_size,3), np.uint8)*255
        croped_img=img[y-offset:y+ h+offset,x-offset:x+ w+offset]
            
        aspect_ratio=h/w
    
        if aspect_ratio>1:
            k=img_size/h
            wCal= math.ceil(k*w)
            img_resize=cv.resize(croped_img,(wCal,img_size))
            wGap =math.ceil((img_size-wCal)/2)
            img_bg[:,wGap:wCal+wGap] = img_resize
            prediction,index=classifier.getPrediction(img_bg)
            # print(labels[index])
            
        else:
            k=img_size/w
            hCal= math.ceil(k*h)
            img_resize=cv.resize(croped_img,(img_size,hCal))
            hGap =math.ceil((img_size-hCal)/2)
            img_bg[hGap:hCal+hGap,:] = img_resize
            prediction,index=classifier.getPrediction(img_bg)
            # print(labels[index])

        cv.putText(img_out,labels[index],(x,y-20),cv.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)
            
        # cv.imshow("Image_croped",croped_img)
        # cv.imshow("Image_bg",img_bg)
        index_l=index
    
    if ret:    
        ret,jpeg=cv.imencode('.jpg',img_out)
        return jpeg.tobytes(),index_l
    else:
        return None


'''
Copyrigths:
Devanand
Dhinesh
'''    