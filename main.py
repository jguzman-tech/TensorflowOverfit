# This Python file uses the following encoding: iso-8859-1

import argparse
import pdb # use pdb.set_trace() to set a "break point" when debugging
import os, sys
import numpy as np
import sys
import scipy
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os.path
import copy
import warnings
import statistics
from matplotlib.pyplot import figure
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# warnings.simplefilter('error') # treat warnings as errors
figure(num=None, figsize=(30, 8), dpi=80, facecolor='w', edgecolor='k')
matplotlib.rc('font', size=24)

def Baseline(X_mat, y_vec, X_new):
    pred_new = np.zeros((X_new.shape[0],))
    if (y_vec == 1).sum() > (y_vec.shape[0] / 2):
        pred_new = np.where(pred_new == 0, 1, pred_new)
    return pred_new

## Load the spam data set and Scale the input matrix
def Parse(fname):
    all_rows = []
    with open(fname) as fp:
        for line in fp:
            row = line.split(' ')
            all_rows.append(row)
    temp_ar = np.array(all_rows, dtype=float)
    temp_ar = temp_ar.astype(float)
    for col in range(temp_ar.shape[1] - 1): # for all but last column (output)
        std = np.std(temp_ar[:, col])
        if(std == 0):
            print("col " + str(col) + " has an std of 0")
        temp_ar[:, col] = stats.zscore(temp_ar[:, col])
    np.random.shuffle(temp_ar)
    return temp_ar

parser = argparse.ArgumentParser(description='Create three NN models with TensorFlow')
args = parser.parse_args()
temp_ar = Parse("spam.data")
X = temp_ar[:, 0:-1] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, -1]]).T 
y = y.astype(int)
num_row = X.shape[0]

## Divide the data into 80% train, 20% test observations (out of all observations in the whole data set).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Next divide the train data into 50% subtrain, 50% validation.
X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(X_train, y_train, test_size=0.5, random_state=0)

## Define a for loop over regularization parameter values, and fit a neural network for each.
# define hidden units
sequence = np.arange(1,11)
hidden_units_vec = np.power(2, sequence)

# create a min loss value array
min_loss_val_arr = list()
cor_loss_train_arr = list()

#loop through hidden_units_vec
for hidden_units in hidden_units_vec:

    model_one = keras.Sequential([
        keras.layers.Flatten(input_shape=X_subtrain[0].shape),
        keras.layers.Dense(hidden_units, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # compiler model
    model_one.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    # feed the model
    results = model_one.fit(X_subtrain, y_subtrain, validation_data = (X_validation, y_validation), epochs=100)

    # choose the min validation loss and tran loss
    min_val_loss = min(results.history['val_loss'])
    cor_train_loss = min(results.history['loss'])

    # put this value to our min_val_loss_arr
    min_loss_val_arr.append(min_val_loss)
    cor_loss_train_arr.append(cor_train_loss)

##On the same plot, show the logistic loss as a function of the regularization parameter
plt.xlabel("hidden units")
plt.ylabel("logistic loss")
plt.plot([i for i in hidden_units_vec], [j for j in cor_loss_train_arr], color="lightblue", linestyle="solid", linewidth=3, label=" subtrain")
plt.plot([i for i in hidden_units_vec],  [j for j in min_loss_val_arr], color="lightblue", linestyle="dashed", linewidth=3, label=" validation")

min_hidden_val_loss = min(min_loss_val_arr)
index_min = min_loss_val_arr.index(min_hidden_val_loss)
best_parameter_value = hidden_units_vec[index_min]
print(f"best_parameter_value = {best_parameter_value}")

plt.scatter(best_parameter_value, min_hidden_val_loss, marker='o', color='red', s=160, facecolor='none', linewidth=3, label='best hidden unites')

plt.legend()
plt.savefig(f"plot.png")

## Re-train the network on the entire train set
print("RE-TRAINING")
# setup layers

retrain_model = keras.Sequential([
    keras.layers.Flatten(input_shape=X_subtrain[0].shape),
    keras.layers.Dense(best_parameter_value, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])

# compiler model
retrain_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# feed the model
# you can access 'loss' from results_one.history['loss']
retrain_result = retrain_model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=100)

print(f"best_parameter_value test accuracy: {retrain_result.history['val_acc'][-1]}")

y_hat =  Baseline(X_train, y_train[:, 0], X_test)
baseline_accuracy = 100 * (np.mean(y_hat == y_test[:, 0]))
print(f"baseline accuracy: {baseline_accuracy}")
