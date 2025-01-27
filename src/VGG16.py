#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:12:34 2024

@author: jonathansetiawan
"""
import os
import cv2
import numpy as np
import glob
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import to_categorical

# Define constants
input_size = [224, 224]
train_image_path = './data/Training_set'
val_image_path = './data/Validation_set'

# Load the VGG16 model without the top layers
vgg = VGG16(input_shape=input_size + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False

# Define a function to read and preprocess images
def read_data(path_image):
    array = []
    label = []
    folders = os.listdir(path_image)
    for i, out_folder in enumerate(folders):
        path = os.path.join(path_image, out_folder)
        image_list = glob.glob(path + '/*.jpg')
        for img in image_list:
            img_cv2 = cv2.imread(img)
            resize_image = cv2.resize(img_cv2, (224, 224))
            array.append(resize_image / 255.0)  # Normalize the image
            label.append(i)  # Assign a label based on the folder index
    x = np.array(array, dtype='float32')
    y = np.array(label, dtype='int')
    return x, y

# Load and preprocess data
x_train, y_train = read_data(train_image_path)
x_val, y_val = read_data(val_image_path)

# Convert labels to one-hot encoding
folders = os.listdir(train_image_path)
num_classes = len(folders)
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

# Add custom layers on top of VGG16
x = Flatten()(vgg.output)
prediction = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=vgg.input, outputs=prediction)

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print model summary
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=40, validation_data=(x_val, y_val))

# Save the model architecture to a JSON file so that it can be tested in the test_code
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights to an HDF5 file
model.save_weights("ckpt.weights.h5")
print("Model saved as model.json and ckpt.weights.h5")