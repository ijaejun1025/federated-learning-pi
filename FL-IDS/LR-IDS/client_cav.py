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
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split
#tf.config.set_visible_devices([], 'CPU')
# Load dataset
TEST_SIZE = 0.33

x, y = utils_cav.load_cav()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=TEST_SIZE,
    shuffle=True,
    stratify=y,
)

# Conv2D expects 4D input: (samples, height, width, channels)
# Here we treat feature vector as height and use width=1, channels=1
if x_train.ndim == 2:
    x_train = x_train[:, :, np.newaxis, np.newaxis]
    x_test = x_test[:, :, np.newaxis, np.newaxis]


# Load model and data (MobileNetV2, CIFAR-10)
model = keras.Sequential([
    keras.layers.Conv2D(96,(4,4),input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]),activation='relu',padding='same'),
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
        
        # Predictions
        y_pred = model.predict(x_test).argmax(axis=1)
        
        # Metrics calculation
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # MAP (Mean Average Precision) calculation
        y_pred_proba = model.predict(x_test)
        map_score = average_precision_score(y_test, y_pred_proba[:, 1])
        
        print(f"Eval accuracy: {accuracy:.4f}, F1: {f1:.4f}, MAP: {map_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        return loss, len(x_test), {
            "accuracy": accuracy,
            "f1_score": f1,
            "map": map_score,
            "precision": precision,
            "recall": recall
        }

# Start Flower client
fl.client.start_numpy_client(
    server_address="192.168.137.68:3040", 
        client=FlowerClient()
)