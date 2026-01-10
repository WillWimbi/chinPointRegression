# import torch
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import cv2
# import random
# import os
# from PIL import Image




# def poen(img):

#     cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     return cv2.resize(img,None,fx=96/256,fy=96/256) 


# def annott():
#     imgs = numpy.zeros((num_imgs,96,96))
#     for image in imageFolder:
#         img = cv.imread(image)
#         img = cv.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         img = cv.resize(img,None,fx=96/256,fy=96/256)
#         imgs.append(img)
#     open ground truth:
#     create numpy array of them. Shape {num_imgs,2}
#     multiply this by 96/256 to get the correct scale.


#     #use batch size of 64:
#     64,96,96 tensors passed in, output = 64,2 sized tensors, GT = 64,2 sized tensors.

# #works for arbitrarily sized images.    
# class chinPointDataset(dataset):
#     def __init__(self, imageFolder, annotFile, idealWidth, idealHeight): #idealsize
#     this.annotFile = annotFile
#     this.imageFolder = imageFolder  
#     this.size = 0  
    
#     with open(imageFolder, 'r') as f:
#         for image in os.listdir(imageFolder):
#             if(image.endswith(".png")):
#                 this.size+=1

#     #open json:
#     with json.open(annotFile, 'r') as f:
#     now have it as json object=jsonAnnot
    


#     this.allImgsArray = numpy.zeroes([count,idealWidth,idealHeight])
#     this.allAnnotsArray = numpy.zeroes([count,2]) 
#     with open(imageFolder, 'r') as f:
#         for idx,image in zip(os.listdir(imageFolder)):
#             if(image.endswith(".png")):            
#                 imgArr = cv2.imread(os.path.join(imageFolder,image))
#                 height,width,channels = imgArr.shape
#                 imgArr = cv.cvtColor(imgArr,cv2.COLOR_BGR2GRAY)
#                 imgArr = cv.resize(imgArr,None,fx=idealWidth,fy=idealHeight)
#                 this.allAnnotsArray[idx] = [jsonAnnot[idx][0]*(idealWidth/width),jsonAnnot[idx][1]*(idealHeight/height) #(each entry is in format "xxxxxx.png", starting at 000001))
          



    
        
                    
#     def __len__():
#         return size

#     def getitem__():
#         get the pickings, and then sample the same indices from the allImgsAray and from the allAnnotsArray
#         sample from this.allImgsArray with batch amount as usual, and 


#     #general architectures:
#     heatmaps work well for regression because they allow you to not have to condense so much information into a regressed
#     or a few regressed points, from my understanding.
#     At the same time, regression still can work in such cases. The typical style is often to have a regression 'head' - 
#     A base network architecture which does not change very much, and a final adjustment depending on the goal.
#     This has enabled numerous varying uses of the same underlying network, often with the initial weights actually frozen and 
#     only the weights of the 'head' trained. 
#     Therefore Resnets have proven to remarkably generalizeable for image tasks, whether they be regression, classification, heatmap prediction,
#     pose estimation, segmentation....

#     residual skip connections allow us to give each further layer the information from a previous one while allowing
#     it to preserve its own changes and use that residual data however it sees fit (however the massive chain rule operating throughout
#     this entire function sees fit, that is in) order to minimize the loss. 
#     Typically the pattern is: conv, activation, batchnorm?
#     however


# def throughmoDled



# class chinPointNet(nn):

#     def __init__(super):
        

#     def forward():
#         nn.Conv2d(96,96)#how is the input size managed here?
#         nn.batchnorm2d --> i suppose this for anything of the form batch,length,width etc
#         nn.ReLU()
    

# def lrScheduler(lr,i):
#     return lr*10**(-i/500) #scales down, 10% of orig at 500 iters, 1% of orig at 1000 etc 
    
# optim.SGD = new optimizer

# imgFold = "./Imgs"
# annF = "Imgs_Annotate.json"

# cdataset = new chinPointDataset(pass them in + ideal size = 96,96) 

# dataloader = dataloader(chinPointDataset)

# lr=0.001
# for i in range(1000):
#     batchImg,batchGTAnnot= dataloader.next
#     batchOutAnnot=model(batchImg)
#     criterion or loss (batchOutAnnot,batchGTAnnot)
    
    
    
#     loss.backward()

#     optimizer(lr).step()
#     if(i%10):
#         print(f"Loss: {f}")
#     lrScheduler(lr)... hmm.....

    
# how to open files - what actually happens? Some method for reading bits, ones and zeroes, is used.
# If the file is text, then its interpreted as text perhaps.

# if we were to do -->
# with open('file.txt') as f:
#     f.read()
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import cv2
import json
import os
import time #for tracking
from torch.utils.data import Dataset, DataLoader

class chinPointDataset(Dataset):
    def __init__(self, imageFolder, annotFile, idealWidth, idealHeight):
        self.imageFolder = imageFolder
        self.annotFile = annotFile
        self.idealWidth = idealWidth
        self.idealHeight = idealHeight
        t1=time.time()
        # count images
        self.size = 0
        for image in os.listdir(imageFolder):
            if image.endswith(".png"): #redundant
                self.size += 1
        
        # open json
        with open(annotFile, 'r') as f:
            self.jsonAnnot = json.load(f)
        
        # preallocate arrays
        self.allImgsArray = np.zeros([self.size, idealHeight, idealWidth], dtype=np.float32)
        self.allAnnotsArray = np.zeros([self.size, 2], dtype=np.float32)
        
        # load all images and annotations

        # loop through and load all iamges in a folder. 
        #create a numpy array from these images.
        #then recolor it and resize it and app
        t2=time.time()
        for idx, image in enumerate(os.listdir(imageFolder)):
            if image.endswith(".png"): #redundant
                
                imgArr = cv2.imread(os.path.join(imageFolder, image))
                height, width, channels = imgArr.shape
                imgArr = cv2.cvtColor(imgArr, cv2.COLOR_BGR2GRAY)
                imgArr = cv2.resize(imgArr, (idealWidth, idealHeight))  # cv2.resize takes (width, height)
                
                # normalize to 0-1 - crucial because nns generally work best with 0-1 normalized data... 
                self.allImgsArray[idx] = imgArr.astype(np.float32) / 255.0
                
                # scale annotation from original coords to ideal coords
                origX, origY = self.jsonAnnot[image]
                
                self.allAnnotsArray[idx] = [
                    origX * (idealWidth / width),
                    origY * (idealHeight / height)
                ]
        t3=time.time()
        print(f"Time taken for each stage:")
        print(f"Time taken for counting & loading annotations: {t2-t1} seconds")
        print(f"Time taken for loading & processing images: {t3-t2} seconds")
        print(f"Total time taken: {t3-t1} seconds")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # returns (image, annotation) as tensors
        img = torch.from_numpy(self.allImgsArray[idx]).unsqueeze(0)  # shape: [1, H, W] for single channel
        annot = torch.from_numpy(self.allAnnotsArray[idx])  # shape: [2]
        return img, annot

class chinPointNet(nn.Module):
    def __init__(self,ks=3):
        super().__init__()
        # simple conv stack -> regression head
        # input: [batch, 1, 96, 96]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=ks, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=ks, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=ks, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # after 3 pools: 96 -> 48 -> 24 -> 12
        # so final feature map is [batch, 128, 12, 12]
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 2)  # output: [batch, 2] for (x, y)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def lrScheduler(baseLr, i):
    # formula: scales down, 10% of orig at 500 iters, 1% at 1000
    return baseLr * (10 ** (-i / 500))

imageFolder = "Imgs"
annotFile = "Imgs_Annotate.json"
idealWidth = 96
idealHeight = 96
chinPointDataset = chinPointDataset(imageFolder, annotFile, idealWidth, idealHeight)

dataloader = DataLoader(
    chinPointDataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,              # ✅ Parallel data loading
    pin_memory=True,            # ✅ CRITICAL for fast GPU transfer!
    persistent_workers=True     # ✅ Reuse workers across epochs
)

# Initialize model ON GPU
model = chinPointNet().to("cuda") 

# Loss function (can stay on CPU, will auto-move)
criterion = nn.MSELoss()

# Optimizer
baseLr = 0.001
optimizer = optim.SGD(model.parameters(), lr=baseLr)  # Added momentum


