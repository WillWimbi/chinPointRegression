import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
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
import F24Config as config
import pandas as pd

#import pyautogui

for file in os.listdir(config.OUTPUT_PATH):
    filePath = os.path.join(config.OUTPUT_PATH, file)
    os.remove(filePath)

class ChinPointDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            annotations_file (string): Path to the JSON file with annotations.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir

        with open(annotations_file) as f:
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = list(self.annotations.keys())[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')
        
        chin_point = self.annotations[img_name]
        chin_point = torch.tensor(chin_point, dtype=torch.float32) / (config.INIT_SIZE/96)

        if self.transform:
            image = self.transform(image)
        # print(image.shape)
        # print(chin_point)
        return image, chin_point

class ChinPointModel(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        #self.batchnorm1  = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        #self.batchnorm2  = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        #self.batchnorm3  = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 2) 
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = self.batchnorm1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        #x = self.batchnorm2(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        #x = self.batchnorm3(x)
        x = self.pool(x)
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        out = self.fc1(x) 
        return out


transform = transforms.Compose([
    transforms.Resize((config.DEF_SIZE,config.DEF_SIZE)),
    transforms.ToTensor(),
    
])

#def val_loss():

dataset = ChinPointDataset(config.ROOT_PATH,config.ANNOT_PATH,transform=transform)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
dataset_length = dataset.__len__()
train_steps = dataset_length//config.BATCH_SIZE

#split data

model = ChinPointModel().to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LR)
loss_avg = []
LR_avg = []
t1=time.time()
criterion = nn.MSELoss()
forward_hooks = []
backward_hooks = []

for j in range(config.EPOCHS):
    loss_interval = 0
    for i in range(train_steps):
        
        optimizer.zero_grad()
        citer = j*train_steps + i +1 ###adding 1!!!
        LR = config.get_lr(citer,config.LR,250,halfway=0,train_steps=0)
        #this is all just for watching gradients.
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR

        if citer==1:
            layer_outputs = []
            def capture_output(module, input, output):
                layer_outputs.append(output.to('cpu').detach())
            for name, layer in model.named_modules():
                if isinstance(layer, nn.Conv2d):  # or any other layer type you're interested in
                    hook = layer.register_forward_hook(capture_output)
                    forward_hooks.append(hook)
            
            grads = []
            def capture_grads(module, input, grad):
                grads.append(grad[0].to('cpu').detach())
            
            for name, layer in model.named_modules():
                if isinstance(layer, nn.Conv2d):  # or any other layer type you're interested in
                    hook = layer.register_full_backward_hook(capture_grads)
                    backward_hooks.append(hook)
        #end of capturing gradients.

        data, nlabel = next(iter(dataloader))
        data = data.to(config.DEVICE)
        nlabel = nlabel.to(config.DEVICE)
        label=nlabel.view(-1)
        noutput = model(data)
        output = noutput.view(-1)
        print('label \n',label)
        print('output \n',output)
        print('label size \n',label.size())
        print('output size\n',output.size())
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print(f"Epoch = {j}/{config.EPOCHS} - Step {citer}: Loss = {loss.item()} --> LR = {LR}")
        if 3<citer<8 or 27<citer<32 or 55<citer<60 or 95<citer<100 or 195<citer<200 or 595<citer<600 or 995<citer<1000 or 3995<citer<4000 or 6995<citer<7000 or 11995<citer<12000 or 16995<citer<17000:
            config.savePreds(noutput,nlabel,citer,data)
        loss_interval += math.log(loss.item())/train_steps
        LR_avg.append(LR)
        if citer%999==0 or citer==499:
            plt.plot(loss_avg)
            plt.savefig(f'.outputs\loss_{i}.png')
            plt.close()
        # if citer%10==0:
        #     dat = (data[0].to('cpu')).squeeze(0).numpy()
        #     plt.imshow(dat, cmap='gray' if len(dat.shape) == 2 else None)
        #     plt.axis('off')
        #     plt.show()
        if citer==1:
            for hook in forward_hooks:
                hook.remove()

            for hook in backward_hooks:
                hook.remove()
        #break
    loss_avg.append(round(loss_interval,8))
        
t2=time.time()
print("Secs: ",(t2-t1))  
fig,(ax1,ax2) = plt.subplots(2)
ax1.plot(loss_avg, color='blue')
ax1.set_title("Loss over time")
ax2.plot(LR_avg, color='orange')
ax2.set_title("Learning rate over time")
plt.show()