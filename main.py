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
X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(X_train, y_train, test_size=0.5)

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
    results = model_one.fit(X_subtrain, y_subtrain, validation_data = (X_validation, y_validation), epochs=args.max_epochs_one)
    
    # choose the min validation loss and tran loss
    min_val_loss = min(results.history['val_loss'])
    cor_train_loss = min(results.history['loss'])

    # put this value to our min_val_loss_arr
    min_loss_val_arr.append(min_val_loss)
    cor_loss_train_arr.append(cor_train_loss)

##On the same plot, show the logistic loss as a function of the regularization parameter
plt.xlabel("epoch")
plt.ylabel("logistic loss")
plt.plot([i for i in hidden_units_vec], [j for j in cor_loss_train_arr], color="lightblue", linestyle="solid", linewidth=3, label=" subtrain")
plt.plot([i for i in hidden_units_vec],  [j for j in min_loss_val_arr], color="lightblue", linestyle="dashed", linewidth=3, label=" validation")

best_epochs = results_three.history['val_loss'].index(min_val_loss) + 1

plt.xlabel("epoch")
plt.ylabel("logistic loss")

plt.scatter(best_epochs, min_val_loss, marker='o', color='red', s=160, facecolor='none', linewidth=3, label='1000 h-units minimum')

plt.legend()
plt.savefig(f"{args.max_epochs_one}_{args.max_epochs_two}_{args.max_epochs_three}.png")

# RETRAIN MODELS
# for first model
# setup layers

print("RE-TRAINING")

model_one = keras.Sequential([
    keras.layers.Flatten(input_shape=X_subtrain[0].shape),
    keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])

# compiler model
model_one.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# for second model
# setup layers
model_two = keras.Sequential([
    keras.layers.Flatten(input_shape=X_subtrain[0].shape),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])
# compiler model
model_two.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# for third model
# setup layers
model_three = keras.Sequential([
    keras.layers.Flatten(input_shape=X_subtrain[0].shape),
    keras.layers.Dense(1000, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])
# compiler model
model_three.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# feed the model

# you can access 'loss' from results_one.history['loss']
results_one = model_one.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=best_epochs_one)
results_two = model_two.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=best_epochs_two)
results_three = model_three.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=best_epochs_three)

print(f"10 h-units test accuracy: {results_one.history['val_acc'][-1]}")
print(f"100 h-units test accuracy: {results_two.history['val_acc'][-1]}")
print(f"1000 h-units test accuracy: {results_three.history['val_acc'][-1]}")
y_hat =  Baseline(X_train, y_train[:, 0], X_test)
baseline_accuracy = 100 * (np.mean(y_hat == y_test[:, 0]))
print(f"baseline accuracy: {baseline_accuracy}")
