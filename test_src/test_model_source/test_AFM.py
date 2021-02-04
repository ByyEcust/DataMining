"""
This is a test file and a quick start of AFM based on pytorch
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.14
"""
import torch.nn as nn
from src.model_source.FM import FNN
from src.model_source.AFM import AFM
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
afm_model_ = AFM(dim_features=x_train.shape[1], dim_outputs=1, dim_embedding=64,
                 attention_size=128, dropouts=[0.25, 0.25])

# training framework definition
afm_model = FNN(model=afm_model_, loss_fn=nn.BCELoss())

# training
afm_model.fit(x_train=x_train, y_train=y_train,
              x_valid=x_valid, y_valid=y_valid,
              lr=1e-2, batch_size=128)

# prediction
y_train_pre = afm_model.predict(x_train)
y_valid_pre = afm_model.predict(x_valid)
print('+++ results of AFM +++')
print('the AUC of train: %f' %(roc_auc_score(y_train, y_train_pre)))
print('the AUC of test: %f' %(roc_auc_score(y_valid, y_valid_pre)))
