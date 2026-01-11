import torch
load model weights from checkpoint --> modelCheckpoints/epoch_950.pth

class chinPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stem: 1 -> 32 channels, keep spatial size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 1: 32 -> 64 channels
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.skip2 = nn.Conv2d(32, 64, kernel_size=1)  # 1x1 conv to match channels
        
        # Block 2: 64 -> 128 channels
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.skip3 = nn.Conv2d(64, 128, kernel_size=1)
        
        # Block 3: 128 -> 256 channels
        self.conv4a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(256)
        self.skip4 = nn.Conv2d(128, 256, kernel_size=1)
        
        # Block 4: 256 -> 512 channels
        self.conv5a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5a = nn.BatchNorm2d(512)
        self.conv5b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm2d(512)
        self.skip5 = nn.Conv2d(256, 512, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Adaptive pooling: no matter the spatial size, output is [B, 512, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Head
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 2)
        
        # Initialize final layer small (helps regression converge)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        # Stem: [B, 1, 96, 96] -> [B, 32, 96, 96] -> pool -> [B, 32, 48, 48]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Block 1 with residual: [B, 32, 48, 48] -> [B, 64, 48, 48] -> pool -> [B, 64, 24, 24]
        identity = self.skip2(x)
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = self.relu(x + identity)
        x = self.pool(x)
        
        # Block 2 with residual: [B, 64, 24, 24] -> [B, 128, 24, 24] -> pool -> [B, 128, 12, 12]
        identity = self.skip3(x)
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3b(self.conv3b(x))
        x = self.relu(x + identity)
        x = self.pool(x)
        
        # Block 3 with residual: [B, 128, 12, 12] -> [B, 256, 12, 12] -> pool -> [B, 256, 6, 6]
        identity = self.skip4(x)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.bn4b(self.conv4b(x))
        x = self.relu(x + identity)
        x = self.pool(x)
        
        # Block 4 with residual: [B, 256, 6, 6] -> [B, 512, 6, 6] -> pool -> [B, 512, 3, 3]
        identity = self.skip5(x)
        x = self.relu(self.bn5a(self.conv5a(x)))
        x = self.bn5b(self.conv5b(x))
        x = self.relu(x + identity)
        x = self.pool(x)
        
        # Global average pool: [B, 512, 3, 3] -> [B, 512, 1, 1]
        x = self.avgpool(x)
        
        # Flatten and head: [B, 512] -> [B, 256] -> [B, 2]
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

run model through test dataset: 

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

testChinPointDataset = chinPointDataset(testImageFolder, None, idealWidth, idealHeight)
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


batchTestImgs = next(iter(testDataloader))

savePreds(preds=,label=None,it=0,da=batchTestImgs,def_size=96,outputPath="myTestPredPlots",batchsize=36,printStatements=False)

