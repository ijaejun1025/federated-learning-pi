#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils_cav
#tf.config.set_visible_devices([], 'CPU')
# Load dataset
(x_train, y_train), (x_test, y_test) = utils_cav.load_cav()


# Load model and data (MobileNetV2, CIFAR-10)
model = keras.Sequential([
    keras.layers.Conv2D(96,(4,4),input_shape=(x_train.shape[1],x_train.shape[2],1),activation='relu',padding='same'),
    keras.layers.Conv2D(64,(3,3),activation="relu",padding='same'),
    keras.layers.Conv2D(32,(2,2),activation="relu",padding='same'),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation="relu"),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(32,activation="relu"),
    keras.layers.Dense(2,activation="softmax"),
    
    
    
    ])

model.compile("adam", "sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
model.summary()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="0.0.0.0:3040", 
        client=FlowerClient()
)





