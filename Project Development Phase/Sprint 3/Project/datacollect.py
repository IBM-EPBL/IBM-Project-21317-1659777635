import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import trainlist
import os

def collectData(save_folder):
    cap=cv.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    offset=20
    img_size=300
    counter=0

    while counter<100:
        ret,img=cap.read()
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
            else:
                k=img_size/w
                hCal= math.ceil(k*h)
                img_resize=cv.resize(croped_img,(img_size,hCal))
                hGap =math.ceil((img_size-hCal)/2)
                img_bg[hGap:hCal+hGap,:] = img_resize

            
            cv.imshow("Image_croped",croped_img)
            #img_bw=cv.cvtColor(img_bg,cv.COLOR_BAYER_BG2GRAY)
            cv.imshow("Image_bg",img_bg)
            
        cv.imshow("Image",img)
        key=cv.waitKey(1)
        
        if key==ord("s"):
            counter +=1
            cv.imwrite(f"{save_folder}/Image_{time.time()}.jpg",img_bg)
            print(counter)
        

save_folder="Data/Train_2/"
dataset=trainlist.dataset

for data in dataset:
    data=save_folder+data
    print("\nStarting to Collect "+data)
    try:
        os.mkdir(data)
        collectData(data)
    except:
        continue
    print(data)