import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import cv2
import json
import os
import time #for tracking
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class chinPointDataset(Dataset):
    def __init__(self, imageFolder, annotFile, idealWidth, idealHeight):
        if(annotFile):
            self.imageFolder = imageFolder
            self.annotFile = annotFile
            t1=time.time()
            # count images
            self.size = 0
            for image in os.listdir(imageFolder):
                if image.endswith(".png") or image.endswith(".jpg"): #redundant
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
                if image.endswith(".png") or image.endswith(".jpg"): #redundant
                    
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
            
        else:
            self.annotFile = None
            t1=time.time()
            self.size = len(os.listdir(imageFolder))
            self.allImgsArray = np.zeros([self.size, idealHeight, idealWidth], dtype=np.float32)
            t2=time.time()
            for idx, image in enumerate(os.listdir(imageFolder)):
                if image.endswith(".png"): #redundant
                    imgArr = cv2.imread(os.path.join(imageFolder, image))
                    height, width, channels = imgArr.shape
                    imgArr = cv2.cvtColor(imgArr, cv2.COLOR_BGR2GRAY)
                    imgArr = cv2.resize(imgArr, (idealWidth, idealHeight))  # cv2.resize takes (width, height)
                    
                    # normalize to 0-1 - crucial because nns generally work best with 0-1 normalized data... 
                    self.allImgsArray[idx] = imgArr.astype(np.float32) / 255.0
        t3=time.time()
        print(f"Time taken for each stage:")
        print(f"Time taken for counting & loading annotations: {t2-t1} seconds")
        print(f"Time taken for loading & processing images: {t3-t2} seconds")
        print(f"Total time taken: {t3-t1} seconds")
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        img = torch.from_numpy(self.allImgsArray[idx]).unsqueeze(0)  # shape: [1, H, W] for single channel
        if(self.annotFile!=None):
            # returns (image, annotation) as tensors
            annot = torch.from_numpy(self.allAnnotsArray[idx])  # shape: [2]
            return img, annot
        else:
            return img
        

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
testImageFolder = "testImgs"
idealWidth = 96
idealHeight = 96
trainChinPointDataset = chinPointDataset(imageFolder, annotFile, idealWidth, idealHeight)
testChinPointDataset = chinPointDataset(testImageFolder, None, idealWidth, idealHeight)
dataloader = DataLoader(
    trainChinPointDataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=0,              # ✅ Parallel data loading
    pin_memory=True,            # ✅ CRITICAL for fast GPU transfer!
    persistent_workers=False     # ✅ Reuse workers across epochs
)
testDataloader = DataLoader(
    testChinPointDataset,
    batch_size=36,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)
# Initialize model ON GPU
model = chinPointNet().to("cuda") 

# Loss function (can stay on CPU, will auto-move)
criterion = nn.MSELoss()

# Optimizer
baseLr = 0.001
optimizer = optim.SGD(model.parameters(), lr=baseLr)  # Added momentum


ROOT_PATH = 'Imgs'
ANNOT_PATH = 'Imgs_Annotate.json'
OUTPUT_PATH = 'trainPredPlots'
#BATCH_SIZE = 256
def savePreds(preds, label, it, da, def_size=96,outputPath="trainPredPlots",batchsize=64,printStatements=False):
    # for keys, values in imgAnnot.items():
    #     if keys == imgAnnot
    if printStatements:
        print('saving...')
    
    data = da.to('cpu').squeeze(0).numpy()
    if printStatements:
        print(data.shape)
    data = data.transpose(0,2,3,1)
    # plt.imshow(data[0])
    # plt.plot()
    # plt.show()
    iter = it
    if printStatements:
        print("PREDS:\n",preds)
       # Multiply by def_size and round to nearest integer
    Opreds = torch.round(preds)
    if(label!=None):
        if printStatements:
            print("training da tensor size:",da.shape)
        Olabel = torch.round(label)
    else:
        if printStatements:
            print("testing da tensor size:",da.shape)
        Olabel = None
    #if printStatements:
    #    print("Lookie here: \n",Opreds, Olabel)
    
    # if printStatements:
    #    print("Rounded predictions:", Opreds)
    #    print("Rounded labels:", Olabel)

    fig, axes = plt.subplots(5, 5, figsize=(36, 36))
    if(label!=None):
        fileName = f"{(iter*batchsize):06d}_{((iter+1)*batchsize):06d}_train.png"
    else:
        fileName = f"{(iter*batchsize):06d}_{((iter+1)*batchsize):06d}_test.png"
    for i, ax in enumerate(axes.flat):

        # Directly extract and convert to float from tensors
        xp, yp = Opreds[i][0].item(), Opreds[i][1].item()
        if(label!=None):
            xl, yl = Olabel[i][0].item(), Olabel[i][1].item()
        else:
            xl, yl = None, None
        # if printStatements:
        #    print(xp,yp)
        #    print(xl,yl)

        # if printStatements:
        #    print(f"Coordinates (Preds): x={xp}, y={yp}")
        #    print(f"Coordinates (Label): x={xl}, y={yl}")
        if xp <= 0:
            xp = 0
        if yp <= 0:
            yp = 0
        
        #for file in os.listdir(imgPath):
        #full_path = os.path.join(imgPath, fileName)
        new_path = os.path.join(outputPath, fileName)
        img = data[i]
        ax.imshow(img,cmap='gray')
        ax.axis('off')  # Turn off the axis for a cleaner look
        ax.set_title(f'Image {i + 1}')  # Set the title for each subplot
        
        p = plt.Circle((xp, yp), 3, color='red')
        if(label!=None):
            l = plt.Circle((xl, yl), 3, color='blue')
            ax.add_patch(l)
        ax.add_patch(p)

    plt.tight_layout()
    new_path = os.path.join(outputPath, fileName)
    plt.savefig(new_path)

    #plt.show()
    
    plt.close()

scaler = torch.amp.GradScaler("cuda")  # GradScaler for CUDA AMP

if __name__ == "__main__":
    with open("logs.txt", "a") as f:
        f.write(f"Starting training; date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    # training loop
    for epoch in range(1000):
        for batchIdx, (batchImg, batchGTAnnot) in enumerate(dataloader):
            # forward
            batchImg = batchImg.to("cuda", non_blocking=True)
            batchGTAnnot = batchGTAnnot.to("cuda", non_blocking=True)
            batchOutAnnot = model(batchImg)
            loss = criterion(batchOutAnnot, batchGTAnnot)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                batchOutAnnot = model(batchImg)
                loss = criterion(batchOutAnnot, batchGTAnnot)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        newLr = lrScheduler(baseLr, epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = newLr
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, LR: {newLr:.6f}")
            savePreds(batchOutAnnot,batchGTAnnot,epoch,batchImg)
            testImg = next(iter(testDataloader))
            #quick test to check 
            testImg = testImg.to("cuda", non_blocking=True)
            testOutAnnot = model(testImg)
            savePreds(testOutAnnot,None,epoch,testImg,outputPath="testPredPlots",batchsize=36)
            #logging
            with open("logs.txt", "a") as f:
                f.write(f"epoch: {epoch}; loss: {loss.item()}\n")
            ###########
            # testBatch = next(iter(testDataloader))
            # print(f"Batch shape: {testBatch.shape}")
            # print(f"Min: {testBatch.min()}, Max: {testBatch.max()}")

            # # Visualize directly
            # fig, axes = plt.subplots(6, 6, figsize=(12, 12))
            # for i, ax in enumerate(axes.flat):
            #     ax.imshow(testBatch[i].squeeze().cpu().numpy(), cmap='gray')
            #     ax.axis('off')
            # plt.show()
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"modelCheckpoints/epoch_{epoch}.pth")