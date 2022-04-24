import keras.callbacks

import model
import numpy as np
import random
from model import *
from keras import backend as keras
from data import *
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras import *
from tqdm import tqdm
from glob import glob

IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3

TRAIN_PATH = "unet_data"
TEST_PATH = "test_data"


train_ids_img = sorted(glob(os.path.join(TRAIN_PATH, "images/*")))
train_ids_msk = sorted(glob(os.path.join(TRAIN_PATH, "masks/*")))
test_ids = sorted(glob(os.path.join(TEST_PATH, "images/*")))

# Initialisation of the training
X_train = np.zeros((len(train_ids_img), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids_img), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)


def filler(input_data, arr, img_not_msk):
    for n, id_ in tqdm(enumerate(input_data), total=len(input_data)):
        # filling the training array with the data of each image/mask
        if img_not_msk:
            img = imread(id_)[:, :, :IMG_CHANNELS]
        else:
            img = imread(id_)[:, :, :1]
        # Resizing for the edges cases (res smaller than 64*64)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        arr[n] = img


print("reading the training images")
filler(train_ids_img, X_train, True)
print("reading the training masks")
filler(train_ids_msk, Y_train, False)
print("reading the training masks")
filler(test_ids, X_test, True)
print("Done!")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    unetModule = CreateUnetModule(None, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

    callbacks = [keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
                 keras.callbacks.ModelCheckpoint('model_for_nails.h5', verbose=1, save_best_only=True),
                 keras.callbacks.TensorBoard(log_dir='logs')]
    # logs is the name of dir we're saving in
    ###########################

    results = unetModule.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
