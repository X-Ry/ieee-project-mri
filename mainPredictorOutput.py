#based on: https://www.kaggle.com/code/monkira/brain-mri-segmentation-using-unet-keras/notebook

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from skimage.color import rgb2gray
from keras import Input
from keras.models import Model, load_model, save_model
from keras.models import model_from_json
from keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, \
    MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

from PIL import Image
from os.path import exists


# Set Parameters, Import Data
im_width = 256
im_height = 256

#Define loss function & metrics so we can reload the model

smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)

#load the model for classification

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('model.h5')

def predictor(inputImagePath):

    #image preprocessing
    im = Image.open(inputImagePath)
    im.save('currentBrain.tif')

    img = cv2.imread('currentBrain.tif')
    img = cv2.resize(img ,(28, 28))
    img = img[np.newaxis, :, :, :]

    #batch stuff
    np.expand_dims(img, axis=0)

    #prediction
    y_predicted=loaded_model.predict(img)
    y_predicted=np.argmax(y_predicted, axis=1)
    return y_predicted

#load the model for images

image_model = load_model('unet_brain_mri_seg_official.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

#the prediction function for images

def img_predictor(inputImagePath):

    #image preprocessing
    im = Image.open(inputImagePath)
    im.save('currentBrain.tif')

    img = cv2.imread('currentBrain.tif')
    img = cv2.resize(img ,(im_height, im_width))
    img = img / 255
    img = img[np.newaxis, :, :, :]

    #prediction
    pred=image_model.predict(img)

    print(np.average(np.squeeze(pred)))

    plt.subplot(1,2,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.axis('off')
    plt.grid(visible=None)
    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(pred) > .5, cmap = 'gray' )
    plt.title('Prediction')
    plt.axis('off')
    plt.grid(visible=None)
    #plt.show()
    plt.savefig("PredictionComparison.png",bbox_inches='tight')

    plt.clf() 
    #plt.figure(figsize=(12,12))
    plt.imshow(np.squeeze(pred) > .5, cmap = 'gray' )
    plt.axis('off')
    plt.grid(visible=None)
    plt.savefig("Prediction.png", transparent = True, bbox_inches = 'tight', pad_inches = 0)
    im = Image.open("Prediction.png")


    # Extracting pixel map:
    pixel_map = im.load()
    
    # Extracting the width and height 
    # of the image:
    width, height = im.size
    
    tumorFound = 0
    # taking half of the width:
    for i in range(width):
        for j in range(height):
            # getting the RGB pixel value.
            r, g, b, p = im.getpixel((i, j))
            if r > 0 or g > 0 or b > 0:
                #pixel_map[i, j] = (255,0,0,255)
                tumorFound = 1   
    #im.show()

    if tumorFound:
        print("Found Location of Tumor")
        return [1]
    else:
        print("Did Not Find Location of Tumor")
        return [0]
