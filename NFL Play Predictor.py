# NFL Playcall Predictor
# Based on down, yard line, yards-to-go, time, quarter, score differential, predict whether the next play is
# a run or a pass
# Play-by-play data: 2013-2023, NFLSavant.com

import keras_tuner
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import keras_tuner
from keras import layers
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#%% PBP data import
read_path = os.path.join(os.getcwd() + "\\" + "NFL_PBP_combined.xlsx")
#df_x = pd.read_excel(read_path, usecols= "C:E, H:J, U", header=0, sheet_name="Combined")
df_x_less = pd.read_excel(read_path, usecols= "C:E, H:J", header=0, sheet_name="Combined")
#df_x_less = pd.read_excel(read_path, usecols= "H:I", header=0, sheet_name="Combined")
df_y = pd.read_excel(read_path, usecols= "Y", header=0, sheet_name="Combined")

#df_x_less = tf.convert_to_tensor(df_x_less)
#df_y = tf.convert_to_tensor(df_y)

#%% PBP data from "playground" file
read_path = os.path.join(os.getcwd() + "\\" + "NFL_PBP_combined_playground.xlsx")
df_x_less = pd.read_excel(read_path, usecols= "C, E:G,, I", header=0, sheet_name="Combined_Redux")
df_y = pd.read_excel(read_path, usecols= "H", header=0, sheet_name="Combined_Redux")

#%% Build vanilla neural network
# Rules of thumb: for output, one node per class; softmax activation for multi-class classification
model = keras.Sequential()
#model.add(layers.Dense(384))
model.add(layers.Dense(3))
#model.add(layers.Dense(192, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
#model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
#model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

#opt = keras.optimizers.SGD(learning_rate=0.001)
opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss="SparseCategoricalCrossentropy", metrics='accuracy')

X_train, X_test, y_train, y_test = train_test_split(df_x_less, df_y, test_size=0.33)

#%% Train the model & show results
epochs = 20
batch_sz = 32    # batch size = # of samples until the model is updated; default = 32

history = model.fit(X_train, y_train, batch_size=batch_sz, epochs=epochs, validation_split=0.15)
model.summary()
#TODO: Continue hyperparam tuning

# Plot accuracy during training
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Plot loss during training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


#%% Perform multi-class classification using SVM (SVC)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
X = df_x_less
Y = df_y
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, Y)
Pipeline(steps=[('scaler', StandardScaler()), 'svc', SVC(gamma='auto')])