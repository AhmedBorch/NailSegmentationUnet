import keras.callbacks
import sklearn.model_selection
from keras.losses import mse

import model
import numpy as np
import cv2
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
X = np.zeros((len(train_ids_img), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y = np.zeros((len(train_ids_img), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
# X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

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
filler(train_ids_img, X, True)
print("reading the training masks")
filler(train_ids_msk, Y, False)
# print("reading the images for testing")
# filler(test_ids, X_test, True)
print("Done!")

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

filepath = 'my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
# callbackfile = "saved_models/weights-improvement-{epoch:02}-{val_acc:.2f}.hdf5"



def testchecking(model, X_test, Y_test):

    # make prediction with model
    prediction = model.predict(X_test)
    # print('Model MSE on test data = ', mse(Y_test, prediction).numpy())

    # convert probabilities to integer values
    # predicted_arr = np.argmax(prediction[20], axis=-1)
    predicted_img = np.array(prediction[20] * 255).astype('uint8')
    # grayImage = cv2.cvtColor(predicted_img, cv2.COLOR_GRAY2BGR)

    print("the test pic:")
    testimg = cv2.cvtColor(X_test[20], cv2.COLOR_BGR2RGB)
    cv2.imshow("image", testimg)
    cv2.waitKey()


    # visualize model predictions
    print("the predicted result:")
    cv2.imwrite('gray_img.jpg', predicted_img)
    cv2.imshow("predicted", predicted_img)
    cv2.waitKey()

    print("and now the expected result:")
    cv2.imwrite('gray_img_expec.png', Y_test[20])
    cv2.waitKey()








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    unetModule = CreateUnetModule(None, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    # mode is max because we are monitoring the val_acc (it should be min for val_loss)
    from keras.callbacks import ModelCheckpoint
    callbacks_list = [ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                      mode='min')]
                 # keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
                 #keras.callbacks.TensorBoard(log_dir='logs')

    results = unetModule.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks_list)
    # model.save('fingernailDetectModel.h5')
    # plot the training history
    plt.plot(results.history['loss'], label='Training Loss')
    plt.plot(results.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.savefig('model_training_history')
    plt.show()

    # Load and evaluate the best model version
    model = load_model(filepath)
    results = model.evaluate(X_test, Y_test)
    print("test loss, test acc:", results)
    # val_preds = model.predict(X_test)
    # print(val_preds[0])
    # testchecking(model, X_test, Y_test)
    yhat = model.predict(X_test)
    print('Model MSE on test data = ', mse(Y_test, yhat).numpy())

