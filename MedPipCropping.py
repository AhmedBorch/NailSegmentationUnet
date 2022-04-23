import cv2
import mediapipe as mp
import os
from data_man import *
from glob import glob
from tqdm import tqdm

def cropper(images, masks, save_path):
    # Getting MediaPipe Model
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()#static_image_mode=False,
                          #max_num_hands=2,
                          #model_complexity=0.5,
                          #min_tracking_confidence=0.5,
                          #min_detection_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    #the names of the cropped images and masks
    # image_name = "croppedImg"
    # mask_name = "croppedMask"

    #Padding from the center of the finger tip to get a 64*64 cropped image
    pad = 32
    j = 0
    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extracting the name and extension of the image and the mask. """
        name = x.split("/")[-1].split(".")
        image_name = name[0].split("\\")[-1]
        image_folder = name[0].split("\\")[-2]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name = name[0].split("\\")[-1]
        mask_folder = name[0].split("\\")[-2]
        mask_extn = name[1]

        """ Reading image and mask. """
        img = cv2.imread(x, cv2.IMREAD_COLOR)
        msk = cv2.imread(y, cv2.IMREAD_COLOR)
        # img = cv2.imread(os.path.join(path, "images", "IMG12.jpg"))
        # msk = cv2.imread(os.path.join(path, "masks", "Mask12.jpg"))

        """ cropping """
        # hands model only uses RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        #the positions of the finger tips Landmarks are 4, 8, 12, 16, 20
        i = 4

        # lmList keeps the location of the fingertips landmark
        lmList = []
        if results.multi_hand_landmarks:
            # multi_hand_landmarks[0] is for the first hand
            myHand = results.multi_hand_landmarks[0]
            # changing the coordinates to pixels
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

            while i <= 20:
                tipPosition = lmList[i][1:]
                croppedTip = img[tipPosition[1]-pad:tipPosition[1]+pad, tipPosition[0]-pad:tipPosition[0]+pad]
                croppedMask = msk[tipPosition[1]-pad:tipPosition[1]+pad, tipPosition[0]-pad:tipPosition[0]+pad]
                tmp_img_name = f"{image_name}_{i}.{image_extn}"
                tmp_mask_name = f"{mask_name}_{i}.{mask_extn}"

                cv2.imwrite(os.path.join(save_path, image_folder, tmp_img_name), croppedTip)
                cv2.imwrite(os.path.join(save_path, mask_folder, tmp_mask_name), croppedMask)
                i += 4

""" Loading all images and masks. """
path = "new_data"
images, masks = load_data(path)

""" Creating folders for the cropped images and masks """
save_path = "unet_data"
create_dir(os.path.join(save_path, "images"))
create_dir(os.path.join(save_path, "masks"))

""" Cropping the fingertips """
cropper(images, masks, save_path)

