import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import random
import os
from PIL import Image
#pillow:
BGR default Blue,Green,REd



PIL.Image.open()
#cv2:


cv.resize*(img,None,fx=96/256,fy=96/256)

def poen(img):

    cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.resize(img,None,fx=96/256,fy=96/256) 


def annott():
    imgs = numpy.zeros((num_imgs,96,96))
    for image in imageFolder:
        img = cv.imread(image)
        img = cv.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv.resize(img,None,fx=96/256,fy=96/256)
        imgs.append(img)
    open ground truth:
    create numpy array of them. Shape {num_imgs,2}
    multiply this by 96/256 to get the correct scale.


    #use batch size of 64:
    64,96,96 tensors passed in, output = 64,2 sized tensors, GT = 64,2 sized tensors.