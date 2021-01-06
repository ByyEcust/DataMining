"""
This is a test file and a quick start of LR & FM  based on pytorch
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.06
"""


import numpy as np
import pandas as pd
import torch.nn as nn
from src.model_source.FM import LR, FM, FNN
from sklearn import datasets
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
lr_model_ = LR(dim_features=x_train.shape[1])
fm_model_ = FM(dim_features=x_train.shape[1], dim_embedding=4)

# training framework definition
lr_model = FNN(model=lr_model_, loss_fn=nn.BCELoss())
fm_model = FNN(model=fm_model_, loss_fn=nn.BCELoss())

# training
# lr_model.fit(x_train=x_train, y_train=y_train,
#              x_valid=x_valid, y_valid=y_valid)

fm_model.fit(x_train=x_train, y_train=y_train,
             x_valid=x_valid, y_valid=y_valid,
             lr=1e-2, batch_size=256)



