'''
CatBoost basic model framework
Available for  regression/bi-classification/multi-classification task
*regression: objective='RMSE' or 'MAE'
*classification: objective='Logloss'
*multi-classification: objective='MultiClass'
methods:
    fit
    predict
    model saving
    model loading
    get feature importance

Author: Ruoqiu Zhang (ECUSTwaterman, waterteam), 2020.12.02
'''

import pandas as pd
import catboost as cat
from catboost import CatBoost

class LgbBasicModel(object):
    def __init__(self, objective, features,
                 n_estimators=100,
                 num_leaves=32,
                 min_data_in_leaf=128,
                 max_depth=6,
                 learning_rate=0.01,
                 early_stopping=10,
                 feature_fraction=1.0,
                 bagging_fraction=1.0,
                 lambda_l2=0,
                 device_type='CPU',
                 nthread=-1,
                 verbose=20,
                 seed=2020,
                 model_file=None, **extra_params):
        if model_file:
            self.model = CatBoost()
            self.model.load_model(model_file, format='cbm')
            self.features = features
        else:
            self.model = None
            self.features = features
            self.params = {}
            if device_type == 'CPU':
                self.params['thread_count'] = nthread
            else:
                self.params['rsm'] = feature_fraction
                self.params['device'] = '0'
            self.params['objective'] = objective
            self.params['n_estimators'] = n_estimators
            self.params['max_leaves'] = num_leaves
            self.params['min_data_in_leaf'] = min_data_in_leaf
            self.params['max_depth'] = max_depth
            self.params['learning_rate'] = learning_rate
            self.params['subsample'] = bagging_fraction
            self.params['random_seed'] = seed
            self.params['l2_leaf_reg'] = lambda_l2
            self.params['n_estimators'] = n_estimators
            self.params['verbose'] = verbose
            self.params['early_stopping_rounds'] = early_stopping
            self.params.update(extra_params)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        if not x_valid:
            valid_sets = (x_train[self.features].values, y_train.values)
        else:
            valid_sets = (x_valid[self.features].values, y_valid.values)
        self.model = CatBoost(self.params)
        self.model.fit(x_train[self.features].values, y_train.values, valid_sets=valid_sets)

    def predict(self, x_test):
        return self.model.predict(x_test[self.features].values)

    def model_saving(self, file_path):
        self.model.save_model(file_path, format='cbm')

    def get_features_importance(self):
        features_importance = pd.DataFrame()
        features_importance['features'] = self.features
        features_importance['importance'] = self.model.get_feature_importance()
        return features_importance


