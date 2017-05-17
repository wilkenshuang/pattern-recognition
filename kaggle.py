# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:31:23 2017

@author: wilkenshuang
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, Activation
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

train=pd.read_csv('kaggle/train.csv')
test_images=(pd.read_csv('kaggle/test.csv').values).astype('float32')
train_images=(train.ix[:,1:].values).astype('float32')
train_labels=(train.ix[:,:1].values).astype('int32')

scale=np.max(train_images)
train_images=train_images/scale
test_images=test_images/scale
mean=np.std(train_images)
train_images -= mean
test_images -= mean
input_dim=train_images.shape[1]
train_labels=to_categorical(train_labels)
num_classes=train_labels.shape[1]

# fix random seed for reproducibility
seed=43
np.random.seed(seed)

model=Sequential()
model.add(Dense(128,input_dim=(28*28)))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history=model.fit(train_images,train_labels,validation_split=0.5,
                  epochs=10,batch_size=16,verbose=2)
history_dict=history.history

predictions = model.predict_classes(test_images, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv("kaggle/results.csv", index=False, header=True)
