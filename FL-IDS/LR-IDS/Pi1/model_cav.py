#!/usr/bin/env python
# coding: utf-8

from tensorflow import keras


def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(96, 4, activation="relu", padding="same"),
        keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        keras.layers.Conv1D(32, 2, activation="relu", padding="same"),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    return model
