#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# loading the dataset
path = "FL-IDS/LR-IDS/cav/"
files = [file for file in glob.glob(path + "**/*.csv", recursive=True)]

# Reading all the csv files into dataframes and putting thoose DFs to one list.

dataset = [pd.read_csv(f) for f in files]
def changecolumn(dataset, AttackType):
    df = pd.read_csv(dataset).sample(frac = 0.05, random_state = 20, replace = False).reset_index(drop=True)
    df.columns = ["Timestamp", "CAN ID", "Byte", "DATA[0]","DATA[1]","DATA[2]","DATA[3]","DATA[4]","DATA[5]","DATA[6]","DATA[7]","AttackType"]
    df['AttackType'] = np.where(df['AttackType'] == 'T',AttackType, 'Normal')
    df.dropna()
    return df


dfDos = changecolumn('cav/DoS_dataset.csv','DoS')
dfFuzzy = changecolumn('cav/Fuzzy_dataset.csv','Fuzzy')
dfGear = changecolumn('cav/gear_dataset.csv','Gear-Spooing')
dfRPM = changecolumn('cav/RPM_dataset.csv','RPM-Spoofing')
frames = [dfDos, dfFuzzy, dfGear, dfRPM]
df = pd.concat(frames)
#print(df.head(10))
#print(df.shape)

# dataset shape

dataset = df.dropna()
print('shape of the data',dataset.shape)
print(dataset['AttackType'].value_counts())

def changecolumntype(df):
    for column in df[['CAN ID', 'DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]']]:
        df[column] = df[column].apply(lambda x: int(str(x), base=16))
    return df

dataset = changecolumntype(dataset)
#print(dataset.dtypes)

import datetime
newdf = dataset.copy(deep = True)
dateformat = "%Y-%m-%d %H:%M:%S.%f"
dataset['Timestamp'] = dataset['Timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(float(x)).strftime(dateformat))
#print(dataset.dtypes)
#dataset.head(10)
# In[3]:

bin_label = pd.DataFrame(dataset.AttackType.map(lambda x:'Normal' if x=='Normal' else 'ATTACK'))
bin_data = dataset.copy()
bin_data['AttackType'] = bin_label
from sklearn.preprocessing import LabelEncoder

LE1 = LabelEncoder()

enc_label = bin_label.apply(LE1.fit_transform)
bin_data['intrusion']= enc_label

# one-hot-encoding for attack label

bin_data = pd.get_dummies(bin_data,columns=['AttackType'],prefix="",prefix_sep="")
bin_data['AttackType']= bin_label

# creating dataframe with only numeric attributes of binary class and encoded label attribute
numeric_col = dataset.select_dtypes(include = 'number').columns

numeric_bin = bin_data[numeric_col]
numeric_bin['intrusion'] = bin_data['intrusion']

bin_data['AttackType'].value_counts()


bin_data.to_csv("cav_anoamly.csv")


# In[4]:


def load_cav():
    
    data = pd.read_csv('cav_anoamly.csv')
    data.reset_index(drop=True)
    numeric_cols = data.select_dtypes(include='number').columns
    X = data[numeric_cols].values
    y = data['AttackType'].values

    # Standardizing the features
    x = StandardScaler().fit_transform(X)
    # Label encoding
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)
   
   # """ Select the 80% of the data as Training data and 20% as test data """
    x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.33, random_state=41, shuffle=True, stratify=y)
    return (x_train, y_train), (x_test, y_test)

