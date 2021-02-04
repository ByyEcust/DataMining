"""
Out Of Bags (OOB) combined with K-Folds Cross-validation framework
Two modes:
    local mode: few training samples since same samples are used to estimate local CV
    submit mode: more training samples
methods:
    fit
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.04
"""


class OOB(object):
    def __init__(self, model, cv_method, mode='local'):
        self.model = model  # model is a class and must has fit & predict methods (pre-defined)
        self.method = cv_method  # cv_method must be a KFolds object from sklearn.model_selection (pre-defined)
        self.mode = mode

    def fit(self, train_data, label_data,):
        pass

    def predict(self):
        pass

    def save(self):
        pass
