#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 00:00:28 2020

@author: demon
"""

###image pre processing


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

#importing file STEP1

image_path='path_of_image'

#display image function
def displaymatplotlib(a,title):
    plt.imshow(a),plt.title(title)
    plt.show()

def displaycv2(a,title):
    cv2.imshow(title,a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

#resizing image STEP2

def resizing():
    img=cv2.imread(image_path)
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print("original_image readed")
    height=640
    width=480
    dim=(height,width)
    # res_img=[]
    res=cv2.resize(img, dim,cv2.INTER_LINEAR)
    # res_img=res_img.append(res)
    # print("resized image")
    # resizedImage=res_img
    # displaymatplotlib(res)
    # displaycv2(res,title="resized image")
    return(res)
    
#denosing image STEP3
def denosie():
    de=resizing()
    denoiseimg=cv2.GaussianBlur(de, (0,0), 2)
    displaycv2(denoiseimg,title="denoise image")
    # displaymatplotlib(denoiseimg,title="denoise image")
    return(denoiseimg)
    
#segmentation of image STEP4
def segmentation():
    img=denosie()
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret,gry=cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # displaymatplotlib(gry,title="segmented image")
    displaycv2(gry, title="segmented image")


# denosie()
segmentation()
    
    
