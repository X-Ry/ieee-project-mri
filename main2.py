#Code based on the tutorial from: https://analyticsindiamag.com/brain-tumor-prediction-through-mri-images-using-cnn-in-keras/
#Data from: https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri

import tensorflow as tf
import os, glob
import cv2
from tqdm.notebook import tqdm_notebook as tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import matplotlib.pyplot as plt
import random

os.chdir(r'C:\Users\great\PycharmProjects\neuroimaging\Training\meningioma_tumor')
# os.chdir(r'C:\Users\great\PycharmProjects\neuroimaging\yes')
X_train = []
y_train = []
for i in tqdm(os.listdir()):
    img = cv2.imread(i)
    img = cv2.resize(img, (28, 28))
    X_train.append(img)
    y_train.append((i[0:1]))

os.chdir(r'C:\Users\great\PycharmProjects\neuroimaging\Training\no_tumor')
# os.chdir(r'C:\Users\great\PycharmProjects\neuroimaging\no')

for i in tqdm(os.listdir()):
    img = cv2.imread(i)
    img = cv2.resize(img, (28, 28))
    X_train.append(img)
    y_train.append('N')

os.chdir(r'C:\Users\great\PycharmProjects\neuroimaging\Testing\meningioma_tumor')
# os.chdir(r'C:\Users\great\PycharmProjects\neuroimaging\yes')
X_test = []
y_test = []
for i in tqdm(os.listdir()):
    img = cv2.imread(i)
    img = cv2.resize(img, (28, 28))
    X_test.append(img)
    y_test.append((i[0:1]))

os.chdir(r'C:\Users\great\PycharmProjects\neuroimaging\Testing\no_tumor')
# os.chdir(r'C:\Users\great\PycharmProjects\neuroimaging\no')

for i in tqdm(os.listdir()):
    img = cv2.imread(i)
    img = cv2.resize(img, (28, 28))
    X_test.append(img)
    y_test.append('N')

# ---- TRAINING ----
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("Shape of an image in X_train: ", X_train[0].shape)
print("Shape of an image in X_test: ", X_test[0].shape)

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
y_train = np.array(y_train)
X_train = np.array(X_train)
y_test = np.array(y_test)
X_test = np.array(X_test)

print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)

m1 = Sequential()
m1.add(BatchNormalization(input_shape=(28, 28, 3)))
m1.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
m1.add(MaxPooling2D(pool_size=2))
m1.add(Convolution2D(filters=64, kernel_size=4, padding='same', activation='relu'))
m1.add(MaxPooling2D(pool_size=2))
m1.add(Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu'))
m1.add(MaxPooling2D(pool_size=2))
m1.add(Convolution2D(filters=128, kernel_size=2, padding='same', activation='relu'))
m1.add(MaxPooling2D(pool_size=2))
m1.add(Dropout(0.25))
m1.add(Flatten())
m1.add(Dense(units=128, activation='relu'))
m1.add(Dense(units=64, activation='relu'))
m1.add(Dense(units=32, activation='relu'))
m1.add(Dense(units=2, activation='softmax'))

m1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model
history = m1.fit(X_train, y_train,
                 epochs=50,
                 validation_data=(X_test, y_test),
                 verbose=1,
                 initial_epoch=0)

# ---- RESULTS ----
# visualize accuracy
m1.evaluate(X_test, y_test)
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.rcParams.update({'font.size': 5})


# Check if predictions are correct or not, make an image grid of the fitted model's preditions and the actual answers
os.chdir(r'C:\Users\great\PycharmProjects\neuroimaging')
#OUTDATED, m1 model doesn't do this anymore
#y_predicted = m1.predict_classes(X_test)
#y_actual = np.argmax(y_test, axis=1)
y_predicted = m1.predict(X_test)
y_actual = np.argmax(y_test,axis=1)
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()
#choose some brains randomly to check out:
b = random.sample(range(1, len(y_actual)), L * W)
for i in np.arange(0, L * W):
    axes[i].imshow(X_test[b[i]])
    axes[i].set_title("Prediction Class ="+str(round(y_predicted[b[i]][1], 5))+", Actual Label = "+str(y_actual[b[i]]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.5)
#plt.show()
plt.savefig('50EpochTest.png')

# Classification Report? Dunno how this works
#from sklearn.metrics import classification_report
#print(classification_report(y_true, y_pred))


# ---- WIP ----

# Make a function that inputs an image and outputs information about it

# Probably figure out classifying instead of just saying "is there a tumor or not"

# Make another model that detects where a tumor is likely to be based on an image (use/find another dataset i guess)