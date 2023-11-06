# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:01:46 2023

@author: attav_yg8iw2r
"""

import pandas as pd
import numpy as np
import os
import sys

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

 
# load model
model = load_model('trained_model.h5')
# summarize model.
model.summary()
# load dataset
dataset = pd.read_csv("features.csv")
# split into input (X) and output (Y) variables
X = dataset.iloc[: ,:-1].values
Y = dataset['labels'].values

# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
print("Data Split")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))