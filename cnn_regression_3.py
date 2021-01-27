#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:32:37 2020

@author: chiara
"""

# USAGE
# python cnn_regression.py --dataset Path of the dataset

# import the necessary packages
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyimagesearch import datasets_2
from pyimagesearch import models2
import numpy as np
import argparse
import locale
import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

#from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
#from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"]="0" #per usare la quarta GPU
# per usare la seconda "1" e così via
# se voglio usare prima e seconda insieme "0,1"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of images")
# add other arguments 
#ap.add_argument("-c", "--checkpoints", required=True,
#	help="path to output checkpoint directory")
#ap.add_argument("-m", "--model", type=str,
#	help="path to *specific* model checkpoint to load")
#ap.add_argument("-s", "--start-epoch", type=int, default=0,
#	help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each istance in the dataset and then load the dataset
print("[INFO] loading images attributes...")
inputPath = os.path.sep.join([args["dataset"], "result.txt"])
#inputPath = os.path.sep.join([args["dataset"], "result_trasl.txt"])
df = datasets_2.load_house_attributes(inputPath)
print(df.head())

# load the images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading images...")
images = datasets_2.load_house_images(df, args["dataset"])
images = images / 255.0

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(df, images, test_size=0.20, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# standardScaler instead of scale in the range [0,1]
#sc = StandardScaler()

#trainAttrX = np.array(trainAttrX).reshape(1,-1)
#testAttrX = np.array(testAttrX).reshape(1,-1)

#sc.fit(trainAttrX[:,-1:])
#trainY = sc.transform(trainAttrX[:,-1:])
#testY = sc.transform(testAttrX[:,-1:])

# find the largest z value(house price) in the training set and also 
# the smallest z value and use it to
# scale our z values (house prices) to the range [0, 1] (will lead to better
# training and convergence)
maxValue = trainAttrX["z"].max()
minValue = trainAttrX["z"].min()
trainY = (trainAttrX["z"] - minValue) / (maxValue -minValue)
testY = (testAttrX["z"] -minValue) / (maxValue -minValue)


#trainY = trainAttrX["z"] / maxValue
#testY = testAttrX["z"] / maxValue


# create our Convolutional Neural Network and then compile the model
# using mean absolute percentage error as our loss, implying that we
# seek to minimize the absolute percentage difference between our
# price *predictions* and the *actual prices*
# 1 is the depth for gs instead of 3 that is for RGB 

model = models2.create_cnn(180, 90, 3, regress=True) #modificato da 64, 64, 3
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_squared_error", optimizer=opt)


# train the model
print("[INFO] training model...")
H=model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY),
	epochs=400, batch_size=16)

# make predictions on the testing data
print("[INFO] predicting z...")
preds = model.predict(testImagesX)

#open a file to write matrix pred and test
file=open("matrix_pred_test_20210114_d750nm_value.txt","w")

# compute the difference between the *predicted* z and the
# *actual* z, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

pred = np.array(preds).reshape(len(preds),1)
test = np.array(testY).reshape(len(testY),1)
matrix = np.hstack((pred, test))
#print(matrix)
np.savetxt('matrix_pred_test_20210114_d750nm.txt', matrix)

file.write("[INFO](maxValue-minValue) {:.2f}\n".format((maxValue-minValue)))
file.write("[INFO](maxValue){:.2f}\n".format((maxValue)))
file.write("[INFO](minValue){:.2f}\n".format((minValue)))

#plot
#plt.hist((pred-test)/test, bins = 5)
#plt.show()

#plt.hist2d(matrix[:,0], matrix[:,1], bins=(5, 5), cmap=plt.cm.jet)
#plt.show()

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)


from sklearn.metrics import mean_squared_error

print('rms (sklearn): ', mean_squared_error(testY,preds))

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. z: {}, std z: {}".format(
	locale.currency(df["z"].mean(), grouping=True),
	locale.currency(df["z"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
file.write("[INFO] mean: {:.2f}%, std: {:.2f}%\n".format(mean, std))
file.close()

N=400
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N),H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N),H.history["val_loss"], label="val__loss")
plt.title("LossPlot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="center right")
plt.savefig("plot_loss0114.png")
