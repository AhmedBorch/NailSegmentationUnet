import os
import cv2
import numpy as np
import prespectiveAug
from data_man import *
from tqdm import tqdm
from skimage.util import random_noise
from albumentations import RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def augment_data(images, masks, save_path, augment=True):
    W = 1280
    H = 960

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0].split("\\")[-1]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name = name[0].split("\\")[-1]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y)

        """ Augmentation """
        if augment == True:
            # Add salt-and-pepper noise to the image.
            noise_img = random_noise(x, mode='s&p', amount=0.01)
            # The above function returns a floating-point image
            # on the range [0, 1], thus we changed it to 'uint8 and from [0,255]
            x1 = np.array(255 * noise_img, dtype='uint8')
            y1 = y

            # aug = RandomRotate90(p=1.0)
            # augmented = aug(image=x, mask=y)
            # x2 = augmented['image']
            # y2 = augmented['mask']
            x2 = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
            y2 = cv2.rotate(y, cv2.ROTATE_90_CLOCKWISE)

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            x6 = prespectiveAug.WrapImg(x, 1)
            y6 = prespectiveAug.WrapImg(y, 1)

            x7 = prespectiveAug.WrapImg(x, 2)
            y7 = prespectiveAug.WrapImg(y, 2)

            x8 = rotate_image(x, 15)
            y8 = rotate_image(y, 15)

            x9 = cv2.blur(x, (5, 5))
            y9 = y

            save_images = [x, x1, x2, x3, x4, x5, x6, x7, x8, x9]
            save_masks = [y, y1, y2, y3, y4, y5, y6, y7, y8, y9]
        else:
            save_images = [x]
            save_masks = [y]
        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            tmp_img_name = f"{image_name}_{idx}.{image_extn}"
            tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"

            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1


""" Loading original images and masks. """
path = "Imgs/data/"
images, masks = load_data(path)
print(f"Original Images: {len(images)} - Original Masks: {len(masks)}")

""" Creating folders. """
create_dir(os.path.join("new_data", "images"))
create_dir(os.path.join("new_data", "masks"))

""" Applying data augmentation. """
augment_data(images, masks, "new_data", augment=True)

""" Loading augmented images and masks. """
images, masks = load_data("new_data")
print(f"Augmented Images: {len(images)} - Augmented Masks: {len(masks)}")