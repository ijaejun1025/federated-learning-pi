#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import flwr as fl
import numpy as np
import sys
import keras
import utils
import time

if __name__ == "__main__":
    # Load MNIST dataset from https://www.openml.org/d/554
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(5)
    (X_train, y_train) = utils.partition(X_train, y_train, 5)[partition_id]


    Model = keras.Sequential([
        

        keras.layers.Conv2D(96,(4,4),input_shape=(100778,93,1),activation='relu',padding='same'),
        keras.layers.Conv2D(64,(3,3),activation="relu",padding='same'),
        keras.layers.Conv2D(32,(2,2),activation="relu",padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(512,activation="relu"),
        keras.layers.Dense(128,activation="relu"),
        keras.layers.Dense(32,activation="relu"),
        keras.layers.Dense(2,activation="softmax"),
    
    
    ])

    Model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    start_time = time.time()
    #Training the model
    Model.fit(X_train, y_train, epochs=5, batch_size=64) 
    Model.summary()

    # Final evaluation of the model
    scores = Model.evaluate(X_validation, Y_validation, verbose=0)
    delta = time.time()- start_time
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("Training time: %.2f sec" % (delta))
    
    
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            preds = model.predict_proba(X_test)
            all_classes = {0,1}
            loss = log_loss(y_test, preds, labels=[0,1])
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:5040", client=MnistClient())


# In[ ]:




