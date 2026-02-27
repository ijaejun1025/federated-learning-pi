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
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

        # Save results after final round (round 3)
        current_round = config.get('server_round', 0)
        if current_round == 3:
            print(f"\n{'='*60}")
            print(f"FINAL ROUND COMPLETE - SAVING RESULTS")
            print(f"{'='*60}\n")

            # Make predictions
            y_pred_proba = model.predict(x_test)
            y_pred = np.argmax(y_pred_proba, axis=1)

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            # Save predictions
            predictions_df = pd.DataFrame({
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba_0': y_pred_proba[:, 0],
                'y_pred_proba_1': y_pred_proba[:, 1]
            })
            predictions_df.to_csv('cnn_prediction_results.csv', index=False)
            print("✅ Saved: cnn_prediction_results.csv")

            # Save confusion matrix
            cm_df = pd.DataFrame({
                'Metric': ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
                'Value': [tn, fp, fn, tp]
            })
            cm_df.to_csv('cnn_confusion_matrix_values.csv', index=False)
            print("✅ Saved: cnn_confusion_matrix_values.csv")

            # Save performance metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'False Positive Rate'],
                'Value': [accuracy, precision, recall, f1, fpr]
            })
            metrics_df.to_csv('cnn_performance_metrics.csv', index=False)
            print("✅ Saved: cnn_performance_metrics.csv")

            # Print summary
            print(f"\n{'='*60}")
            print("PERFORMANCE SUMMARY:")
            print(f"{'='*60}")
            print(f"Accuracy:  {accuracy*100:.2f}%")
            print(f"Precision: {precision*100:.2f}%")
            print(f"Recall:    {recall*100:.2f}%")
            print(f"F1-Score:  {f1*100:.2f}%")
            print(f"FPR:       {fpr*100:.2f}%")
            print(f"{'='*60}\n")

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





