import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import math
import time
import cv2
import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import random
import numpy as np

# constant paths
ROOT_PATH = './Imgs'
ANNOT_PATH = './Imgs_Annotate.json'
OUTPUT_PATH = './outputs'
# learning parameters
BATCH_SIZE = 512
DEF_SIZE = 96
INIT_SIZE = 512
LR = 0.001
EPOCHS = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/test split
TEST_SPLIT = 0.2
# show dataset keypoint plot
SHOW_DATASET_PLOT = True
#Figure plot width and height
BFIG = 5

def get_lr(it,lr,max_steps,halfway=0,train_steps=0):
    x = lr*0.1**(it/max_steps)
    if halfway != 0 and train_steps!=0:
        if it > train_steps/2:            
            x = lr*0.1**((halfway * train_steps)/max_steps)
    return x

def savePreds(preds, label, it, da, def_size=DEF_SIZE):
    # for keys, values in imgAnnot.items():
    #     if keys == imgAnnot
    global OUTPUT_PATH
    global BATCH_SIZE
    print('saving...')
    data = da.to('cpu').squeeze(0).numpy()
    print(data.shape)
    data = data.transpose(0,2,3,1)
    # plt.imshow(data[0])
    # plt.plot()
    # plt.show()
    iter = it
    print("PREDS:\n",preds)
       # Multiply by def_size and round to nearest integer
    Opreds = torch.round(preds)
    Olabel = torch.round(label)
    #print("Lookie here: \n",Opreds, Olabel)
    
    # print("Rounded predictions:", Opreds)
    # print("Rounded labels:", Olabel)

    fig, axes = plt.subplots(BFIG, BFIG, figsize=(36, 36))
    for i, ax in enumerate(axes.flat):
        fileName = f"{i+iter:03d}.png"

    # Directly extract and convert to float from tensors
        xp, yp = Opreds[i][0].item(), Opreds[i][1].item()
        xl, yl = Olabel[i][0].item(), Olabel[i][1].item()
        # print(xp,yp)
        # print(xl,yl)

        # print(f"Coordinates (Preds): x={xp}, y={yp}")
        # print(f"Coordinates (Label): x={xl}, y={yl}")
        if xp <= 0:
            xp = 0
        if yp <= 0:
            yp = 0
        
        #for file in os.listdir(imgPath):
        #full_path = os.path.join(imgPath, fileName)
        new_path = os.path.join(OUTPUT_PATH, fileName)
        img = data[i]
        ax.imshow(img,cmap='gray')
        ax.axis('off')  # Turn off the axis for a cleaner look
        ax.set_title(f'Image {i + 1}')  # Set the title for each subplot
        
        p = plt.Circle((xp, yp), 3, color='red')
        l = plt.Circle((xl, yl), 3, color='blue')
        ax.add_patch(p)
        ax.add_patch(l)

    fileName = f"{(iter*BATCH_SIZE):03d}_{((iter+1)*BATCH_SIZE):03d}.png"
    plt.tight_layout()
    new_path = os.path.join(OUTPUT_PATH, fileName)
    plt.savefig(new_path)

    #plt.show()
    
    plt.close()


def showInitializations(layer_outputs, grads):
    for i, layer_output in enumerate(layer_outputs):
        print(f"Layer {i} output:")
        print(f"  Shape: {layer_output.shape}")
        print(f"  Mean: {layer_output.mean().item():.2f}")
        print(f"  Std: {layer_output.std().item():.2f}")
        print(f"  Min: {layer_output.min().item():.2f}")
        print(f"  Max: {layer_output.max().item():.2f}")
        
    # Create histogram
        hy, hx = torch.histogram(layer_output, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())

    plt.legend([f'Layer {i}' for i in range(len(layer_outputs))])
    plt.title('Activation distribution')
    plt.show()

    for i, grad_ in enumerate(grads):
        print(f"Layer {i} grad:")
        print(f"  Shape: {grad_.shape}")
        print(f"  Mean: {grad_.mean().item():.5f}")
        print(f"  Std: {grad_.std().item():.5f}")
        print(f"  Min: {grad_.min().item():.5f}")
        print(f"  Max: {grad_.max().item():.5f}")
        
    # Create histogram
        gy, gx = torch.histogram(grad_, density=True)
        plt.plot(gx[:-1].detach(), gy.detach())

    plt.legend([f'Layer {i}' for i in range(len(grads))])
    plt.title('Gradient distribution')
    plt.show()