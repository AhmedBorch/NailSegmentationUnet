import keras.callbacks

import model
from model import *
from data import *
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras import *
from tqdm import tqdm

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3

# TRAIN_PATH = ''
# TEST_PATH = ''
#
# train_ids = next(os.walk(TRAIN_PATH))[1] #check this thing
# test_ids = next(os.walk(TEST_PATH))[1]
#
# #read all images and resize them
# X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#
# print('resizing training images')
# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
#     path = TRAIN_PATH + id_
#     img = imread(path + id_ + '.png')[:,:,:IMG_CHANNELS]
#     img = resize(img, (IMG_HEIGHT,IMG_WIDTH), mode='constant', preserve_range=True)
#     X_train[n] = img #filling the training array with the data of each image
# #repeat for test set
# print('Done image upload and resize')

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     unetModule = CreateUnetModule()
#
#     #ModuleCheckpoint
#     checkpointer = keras.callbacks.ModelCheckpoint('model_for_nails.h5', verbose=1, save_best_only=True)
#     callbacks = [
#         keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
#         keras.callbacks.TensorBoard(log_dir='logs')]
#         #logs is the name of dir we're saving in
#     ###########################
#
#     results = unetModule.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
