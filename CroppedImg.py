# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:11:59 2017

@author: gd
"""

from PIL import ImageEnhance
from PIL import Image  
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
'''
folder_path='C:/Users/gd/Desktop/test/'
count=1
for i in os.listdir(folder_path):
    if i!='mask':
        img_path=folder_path+i
        image = Image.open(img_path) 
        #image=cv2.imread(img_path)        
        enh_col = ImageEnhance.Color(image)  
        color = 4
        image_colored = enh_col.enhance(color) 
        image=np.array(image_colored)  
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        #hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        #hsv=cv2.dilate(hsv,kernel)
        #redLower=np.array([140,10,170])
        #redHigher=np.array([180,160,250])
        redLower=np.array([80,150,20])
        redHigher=np.array([200,220,128])
        mask=cv2.inRange(lab,redLower,redHigher)
        plt.imsave('C:/Users/gd/Desktop/test/mask/'+str(count)+'.jpg',mask)
        count+=1
'''
img_path='C:/Users/gd/Desktop/redDigits/441.jpg'
image = Image.open(img_path) 
#image=cv2.imread(img_path)        
enh_col = ImageEnhance.Color(image)  
color = 4
image_colored = enh_col.enhance(color) 
image=np.array(image_colored)  
lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
#hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#hsv=cv2.dilate(hsv,kernel)
#redLower=np.array([140,10,170])
#redHigher=np.array([180,160,250])
redLower=np.array([80,150,20])
redHigher=np.array([200,220,128])
mask=cv2.inRange(lab,redLower,redHigher)
ret,cont,hier=cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

points=[]
res=[]
temp=[]
count=0
for i in cont:
    x,y,w,h=cv2.boundingRect(i)
    if w*h>100:
        points.append([x,y,w,h])
temp=sorted(points,key=lambda value:value[1])
while count<len(temp):
    if count!=len(temp)-1 and temp[count+1][1]-temp[count][1]<40:
        x,y,w,h=temp[count]
        tempx,tempy,tempw,temph=temp[count+1]
        #switch=0
            #switch=w
            #w=tempw
            #tempw=switch
        if y>tempy:
            y=tempx
        if h<temph:
            h=temph
        if x>tempx:
            dis=x-tempx-tempw
            x=tempx
            #w=w+tempw
        else:
            dis=tempx-x-w
        w=w+tempw+dis
        #w=w+tempw+dis
        res.append([x,y,w,h])
        count+=2
    else:
        res.append(temp[count])
        count+=1    
output=[]   
for i in range(len(temp)):
    x,y,w,h=temp[i]
    num=image[y-5:y+h+5,x-5:x+w+5]
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #num=cv2.morphologyEx(num,cv2.MORPH_OPEN,kernel)
    num=cv2.cvtColor(num,cv2.COLOR_BGR2GRAY)
    #num=cv2.GaussianBlur(num,(5,5),1)
    ret,th=cv2.threshold(num,220,255,cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
    th=cv2.medianBlur(th,3)

    output.append(th)
for i in range(len(output)):
    plt.figure(i+1)
    plt.imsave('C:/Users/gd/Desktop/redDigits/train/'+str(i)+'.jpg',output[i])
    #plt.imsave('C:/Users/gd/Desktop/1'+str(i)+'.jpg',mask)

#plt.imsave('C:/Users/gd/Desktop/1.jpg',mask)

#ret,th=cv2.threshold(gray,130,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#redLower=np.array([0,10,100])
#redHigher=np.array([180,160,245])
#mask=cv2.inRange(hsv,redLower,redHigher)
#dilated = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))


#plt.imshow(image)
