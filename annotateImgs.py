import cv2
import json
import os
import pyperclip
import random
import pyautogui
import time
import keyboard
import os
import sys

#paths
rawImgs = "rawImgs"
imgsPath = "Imgs"
detailsPath = "imgLabellingDetails.txt"
annotPath = 'Imgs_Annotate.json'

try:
    with open(annotPath, 'r') as file:
        annotations = json.load(file)
except Exception as e:
    print("Annotation Dict did not exist or was not saved properly: - ",e)
    annotations = {}

try:
    with open(detailsPath, 'r') as file:
        oldDetails = json.load(file)
except Exception as e:
    print("Details Dict did not exist or was not saved properly: - ",e)
    oldDetails = {'rawImgsNum':0, 'outputIter':0}

#print info
details = oldDetails
print(len(annotations))

#defining funcs
def onKeyPressP(keyPress):
    global annotations
    global details    
    global currentFile
    global i
    #currentFile = 
    ##print(f"Key pressed: {keyPress.name}")  # This will show which key is being pressed
    if keyPress.name.lower() == "q":
        print('Program Manually Stopped')
        with open(annotPath, 'w') as f:
            json.dump(annotations, f, indent=4)
        details['rawImgsNum'] = i
        with open(detailsPath, 'w') as f:
            json.dump(details, f, indent=4)
        os._exit(0)
           
keyboard.hook(onKeyPressP)

xG,yG = 0,0
# # Function to handle mouse click events
def click_event(event, x, y, flags, params):
    global xG,yG
    if event == cv2.EVENT_LBUTTONDOWN:  
        print(f"Clicked coordinates: ({x}, {y})")
        #annotations[params].append(x, y)
        #cv2.imshow(params, image)
        xG,yG = x, y

#main loop:
iter = details['outputIter']
i=0
for file in os.listdir(rawImgs):
    i+=1
    if oldDetails['rawImgsNum']>i:
        continue

    xG,yG = 0,0

    full_path = os.path.join(rawImgs, file)
    image = cv2.imread(full_path)
    winName = f"{file}, raw iter: {i}, tentative annot iter: {iter+1}"
    cv2.namedWindow(winName)
    cv2.moveWindow(winName, 1571,116)
    #time.sleep(0.1)
    height, width, channels = image.shape
    cv2.imshow(winName, image)

    print("is running")
    cv2.setMouseCallback(winName, click_event)

    cv2.waitKey(0)

    #updating the annotations and details
    if xG !=0 and yG !=0:
        iter+=1
        new_filename = f"{iter:06d}.png"
        annotations[new_filename] = (xG, yG)
        
        # Construct new filename
        output_path = os.path.join(imgsPath, new_filename)
        
        # Save the resized image
        cv2.imwrite(output_path, image)
    else: 
        print(file,"was skipped!!/ or if you quit, it was not saved as starting point.")
    cv2.destroyAllWindows()
    details['rawImgsNum'] = i
    details['outputIter'] = iter

    # Save the annotations to a JSON file
    with open(annotPath, 'w') as f:
        json.dump(annotations, f, indent=4)

    with open(detailsPath, 'w') as f:
        json.dump(details, f, indent=4)

