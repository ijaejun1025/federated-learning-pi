#!/usr/bin/env python
# coding: utf-8

# In[1]:


from types import new_class
from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import openml
from sklearn.model_selection import train_test_split


# In[2]:


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def change_label(df):
   df.class_attack.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
   df.class_attack.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail',
      'snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L',inplace=True)
   df.class_attack.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
   df.class_attack.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R',inplace=True)

# using standard scaler for normalizing
std_scaler = StandardScaler()
def normalization(df,col):
   for i in col:
       arr = df[i]
       arr = np.array(arr)
       df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
   return df
   
# selecting categorical data attributes
cat_col = ['protocol_type','service','flag']

col = ["duration","protocol_type","service","flag","src-bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num-compromised","root-shell","su-attempted","num_root","num_file_creation","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_error_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","class_attack","class_score"]
url = "KDDTrain+.csv"
dataset = pd.read_csv(url,names=col)
dataset.drop(['class_score'],axis=1,inplace=True)
change_label(dataset)
#selecting numeric attributes columns from data
numeric_col = dataset.select_dtypes(include='number').columns

dataset = normalization(dataset.copy(),numeric_col)
categorical = dataset[cat_col]
categorical = pd.get_dummies(categorical,columns=cat_col)
bin_label = pd.DataFrame(dataset.class_attack.map(lambda x:'normal' if x=='normal' else 'abnormal'))
bin_data = dataset.copy()
bin_data['class_attack'] = bin_label
le1 = preprocessing.LabelEncoder()
enc_label = bin_label.apply(le1.fit_transform)
bin_data['intrusion'] = enc_label
bin_data = pd.get_dummies(bin_data,columns=['class_attack'],prefix="",prefix_sep="") 
bin_data['class_attack'] = bin_label
numeric_bin = bin_data[numeric_col]
numeric_bin['intrusion'] = bin_data['intrusion']
corr= numeric_bin.corr()
corr_y = abs(corr['intrusion'])
highest_corr = corr_y[corr_y >0.5]
highest_corr.sort_values(ascending=True)
numeric_bin = bin_data[['count','srv_serror_rate','serror_rate','dst_host_serror_rate','dst_host_srv_serror_rate','logged_in','dst_host_same_srv_rate','dst_host_srv_count','same_srv_rate']]
numeric_bin = numeric_bin.join(categorical)
# then joining encoded, one-hot-encoded, and original attack label attribute
bin_data = numeric_bin.join(bin_data[['abnormal','normal','class_attack','intrusion']])


# In[3]:


bin_data.to_csv("nslkdd_anoamly.csv")


# In[4]:


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


# In[5]:


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = 93  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


# In[6]:


def load_mnist() -> Dataset:
    """Loads the MNIST dataset using OpenML.

    OpenML dataset link: https://www.openml.org/d/554
    """
    data = pd.read_csv('nslkdd_anoamly.csv')
    data.reset_index(drop=True)
    df =np.array(data)
    X = df[:,:93]
    y = df[:,-1]

    # Standardizing the features
    x = StandardScaler().fit_transform(X)
    # Label encoding
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)
   
   # """ Select the 80% of the data as Training data and 20% as test data """
    x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    return (x_train, y_train), (x_test, y_test)


# In[7]:


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )

