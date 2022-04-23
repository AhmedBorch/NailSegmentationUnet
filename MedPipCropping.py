import cv2
import mediapipe as mp
import os
from glob import glob

mpHands = mp.solutions.hands
hands = mpHands.Hands()#static_image_mode=False,
                      #max_num_hands=2,
                      #model_complexity=0.5,
                      #min_tracking_confidence=0.5,
                      #min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

#path to images and masks
path = "new_data"
img = cv2.imread(os.path.join(path, "images", "col", "IMG12.jpg"))
mask = cv2.imread(os.path.join(path, "masks", "masks", "Mask12.jpg"))

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = hands.process(imgRGB)

# hands model only uses RGB
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = hands.process(imgRGB)

#the positions of the finger tips Landmarks are 4, 8, 12, 16, 20
i = 4

image_name = "croppedImg"
mask_name = "croppedMask"

#Padding from the center of the finger tip to get a 64*64 cropped image
pad = 32


lmList = []
if results.multi_hand_landmarks:
    # multi_hand_landmarks[0] is for the first hand
    myHand = results.multi_hand_landmarks[0]
    for id, lm in enumerate(myHand.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        lmList.append([id, cx, cy])

while i <= 20:
    tipPosition = lmList[i][1:]
    #print (tipPosition)

    #crop_image = img[miny - pad:maxy + pad, minx - pad:maxx + pad]
    croppedTip = img[tipPosition[1]-pad:tipPosition[1]+pad, tipPosition[0]-pad:tipPosition[0]+pad ]
    croppedMask = mask[tipPosition[1]-pad:tipPosition[1]+pad, tipPosition[0]-pad:tipPosition[0]+pad ]
    tmp_img_name = f"{image_name}_{i}.jpg"
    tmp_mask_name = f"{mask_name}_{i}.jpg"

    cv2.imwrite(os.path.join(path, tmp_img_name), croppedTip)
    i += 4
