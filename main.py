import os
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pathlib import Path
from tensorflow.keras.utils import plot_model

import createCNN
import dbcreate

DATA = Path('.') / "Database"
TEST = Path('.') / "Testing"

CATEGORIES = ["analysis", "and", "bad", "bed", "blood", "cotton", "drop", "family"]

training_data = []  # list with pixels and categories tables
testing_data = []  # list with pixels

# numpy arrays as input data
X = []
Y = []
Z = []

# amount of convolutional and dense layers, maximal size of layer in model
dense_layers = [1]
conv_layers = [3]
max_layer_size = [128]

# size of transformed pictures
X_P = 86
Y_P = 86

categories_size = len(CATEGORIES)

# preparation of training data
training_data = dbcreate.create_training_data(X_P, Y_P, DATA, CATEGORIES, training_data)
random.shuffle(training_data)
print(len(training_data))

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, X_P, Y_P, 1)
X = np.array(X, dtype="float") / 255
Y = np.array(Y)

# plt.imshow(X[0], cmap='gray')
# plt.show()

# print(np.shape(X))
# print(np.shape(Y))

# preparation of testing data
testing_data = dbcreate.create_testing_data(X_P, Y_P, TEST, testing_data)
for features in testing_data:
    Z.append(features)

Z = np.array(Z).reshape(-1, X_P, Y_P, 1)
Z = np.array(Z, dtype="float")

# plt.imshow(Z[0], cmap='gray')
# plt.show()

# print(np.shape(Z))

# training and evaluation of model
createCNN.create_neural_network(dense_layers, max_layer_size, conv_layers, X, Y, categories_size)
model = tf.keras.models.load_model("CNN.model")
model.evaluate(X, Y)

# prediction of testing data
prediction2 = model.predict([Z])
# print(prediction2)  # will be a list in a list.

# graphical representation of model
dot_img_file = Path('.') / "picmodel/model_1.png"
plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

# simple functions to check amount of correct predictions
catList = []
counter = 0.0
k = 0
for i in prediction2:  # create list of all results
    flag = 0
    for j in range(len(CATEGORIES)):
        if i[j] == 1:
            catList.append(CATEGORIES[j])
            flag = 1
            # print(CATEGORIES[j])
    if flag == 0:
        catList.append("null")  # result for single picture doesn't contain "1" - discard for not being sure of result

# compare list with picture names, count and print good results percentage
i = 0
for k in os.listdir(TEST):
    print(k[:-10] + " " + catList[i] + " " + str(counter + 1))
    if k[:-9] == catList[i]:
        counter = counter + 1
    if k[:-10] == catList[i]:
        counter = counter + 1
    i = i + 1

print(str(counter) + " / " + str(len(os.listdir(TEST))) + " Percent: " + str(counter / len(os.listdir(TEST)) * 100))
