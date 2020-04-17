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
parser.add_argument('max_epochs_one', type=int,default=300,
                    help='maximum epochs for the 10 hidden units model')
parser.add_argument('max_epochs_two', type=int,default=300,
                    help='maximum epochs for the 100 hidden units model')
parser.add_argument('max_epochs_three', type=int,default=300,
                    help='maximum epochs for the 1000 hidden units model')
args = parser.parse_args()
temp_ar = Parse("spam.data")
X = temp_ar[:, 0:-1] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, -1]]).T 
y = y.astype(int)
num_row = X.shape[0]

## Divide the data into 80% train, 20% test observations (out of all observations in the whole data set).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Next divide the train data into 60% subtrain, 40% validation.
X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(X_train, y_train, test_size=0.4)

## Define three different neural networks
# for first model
# setup layers

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
results_one = model_one.fit(X_subtrain, y_subtrain, validation_data = (X_validation, y_validation), epochs=args.max_epochs_one)
results_two = model_two.fit(X_subtrain, y_subtrain, validation_data = (X_validation, y_validation), epochs=args.max_epochs_two)
results_three = model_three.fit(X_subtrain, y_subtrain, validation_data = (X_validation, y_validation), epochs=args.max_epochs_three)

min_val_loss_one = min(results_one.history['val_loss'])
min_val_loss_two = min(results_two.history['val_loss'])
min_val_loss_three = min(results_three.history['val_loss'])

best_epochs_one = results_one.history['val_loss'].index(min_val_loss_one) + 1
best_epochs_two = results_two.history['val_loss'].index(min_val_loss_two) + 1
best_epochs_three = results_three.history['val_loss'].index(min_val_loss_three) + 1

plt.xlabel("epoch")
plt.ylabel("logistic loss")

plt.plot([i for i in range(1, args.max_epochs_one + 1)], results_one.history['loss'], color="lightblue", linestyle="solid", linewidth=3, label="10 h-units subtrain")
plt.plot([i for i in range(1, args.max_epochs_one + 1)], results_one.history['val_loss'], color="lightblue", linestyle="dashed", linewidth=3, label="10 h-units validation")

plt.plot([i for i in range(1, args.max_epochs_two + 1)], results_two.history['loss'], color="darkblue", linestyle="solid", linewidth=3, label="100 h-units subtrain")
plt.plot([i for i in range(1, args.max_epochs_two + 1)], results_two.history['val_loss'], color="darkblue", linestyle="dashed", linewidth=3, label="100 h-units validation")

plt.plot([i for i in range(1, args.max_epochs_three + 1)], results_three.history['loss'], color="black", linestyle="solid", linewidth=3, label="1000 h-units subtrain")
plt.plot([i for i in range(1, args.max_epochs_three + 1)], results_three.history['val_loss'], color="black", linestyle="dashed", linewidth=3, label="1000 h-units validation")

plt.scatter(best_epochs_one, min_val_loss_one, marker='+', color='red', s=160, facecolor='red', linewidth=3, label='10 h-units minimum')
plt.scatter(best_epochs_two, min_val_loss_two, marker='x', color='red', s=160, facecolor='red', linewidth=3, label='100 h-units minimum')
plt.scatter(best_epochs_three, min_val_loss_three, marker='o', color='red', s=160, facecolor='none', linewidth=3, label='1000 h-units minimum')

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
