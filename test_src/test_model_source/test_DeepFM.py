"""
This is a test file and a quick start of DeepFM  based on pytorch
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.14
"""
import torch.nn as nn
from src.model_source.FM import FNN
from src.model_source.DeepFM import DeepFM
from sklearn import datasets
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# test data generation
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2020)

# data preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)

# model definition
deepFM_model_ = DeepFM(num_inputs=x_train.shape[1], num_outputs=1,
                       num_embedding=32,
                       hidden_size=(128, 128))

# training framework definition
deepFM_model = FNN(model=deepFM_model_, loss_fn=nn.BCELoss())

# training
deepFM_model.fit(x_train=x_train, y_train=y_train,
                 x_valid=x_valid, y_valid=y_valid,
                 lr=1e-2, batch_size=256)

# prediction
y_train_pre = deepFM_model.predict(x_train)
y_valid_pre = deepFM_model.predict(x_valid)
print('+++ results of DeepFM +++')
print('the AUC of train: %f' %(roc_auc_score(y_train, y_train_pre)))
print('the AUC of test: %f' %(roc_auc_score(y_valid, y_valid_pre)))
