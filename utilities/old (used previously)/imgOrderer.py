
import pyperclip
import random
import pyautogui
import time
import keyboard
import os

def logFile(fpath,style,content):
    with open(fpath,style) as f:
        f.write(content)

def click():
    pyautogui.moveTo(2010,-1128, duration=0.1)
    #time.sleep(3)
    pyautogui.leftClick()
    pyautogui.leftClick()
    pyautogui.hotkey('ctrl','a')
    pyautogui.hotkey('ctrl','v')
    time.sleep(.1)
    pyautogui.moveTo(2842,-613, duration=0.1)
    pyautogui.leftClick()

def onKeyPress(keyPress):
    ##print(f"Key pressed: {keyPress.name}")  # This will show which key is being pressed
    global WaitTime
    global Name
    if keyPress.name.lower() == "q":
        print('Program Manually Stopped')
        print('Last name:',Name[2:6])

        logFile('WARNING.txt','w',Item)
        os._exit(0)
    elif keyPress.name.lower() == "p" and WaitTime == 0:
        WaitTime=4
    elif keyPress.name.lower() == "o" and WaitTime == 4:
        WaitTime=0
WaitTime = 0
keyboard.hook(onKeyPress)
stopNum = 108
dataDir = "textDescr"
Item = None
Name = ""
lenDataDir = os.listdir(dataDir)
def orderImgs(folder_path): 
    global stopNum
    global Name
        # Loop through all items in the directory
    for item in os.listdir(folder_path):
        Name = list(item)
        if int("".join(Name[2:6]))<stopNum:

            continue
        global Item
        Item = item
        # Get the full path to the item
        item_path = os.path.join(folder_path, item)
        # Open and read the file
        with open(item_path, 'r') as file:
            content = file.read()
        print(content) #prints content of file
        pyperclip.copy(content)
        click()
        #print("Cycle complete, sleeping...")  # Confirm each cycle
        time.sleep(WaitTime)
orderImgs(dataDir)


# Generate and print a prompt

    
