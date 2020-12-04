'''
LightGBM basic model framework
Available for  regression/bi-classification/multi-classification task
*regression: objective='regresssion'
*classification: objective='binary'
*multi-classification: objective='multiclass' | extra_params['num_class'] = K
methods:
    fit
    predict
    model saving
    model loading
    get feature importance

Author: Ruoqiu Zhang (ECUSTwaterman, waterteam), 2020.12.02
'''

import pandas as pd
import lightgbm as lgb


class LgbBasicModel(object):
    def __init__(self, objective, features, cate_idx=None,
                 n_estimators=100,
                 num_leaves=32,
                 min_data_in_leaf=128,
                 max_depth=6,
                 learning_rate=0.01,
                 early_stopping=10,
                 feature_fraction=1.0,
                 bagging_fraction=1.0,
                 lambda_l2=0,
                 nthread=-1,
                 verbose=20,
                 seed=2020,
                 model_file=None, **extra_params):
        if model_file:
            self.model = lgb.Booster(model_file=model_file)
            self.features = features
        else:
            self.model = None
            self.features = features
            self.cate_idx = cate_idx
            self.params = {}
            self.params['objective'] = objective
            self.params['num_leaves'] = num_leaves
            self.params['min_data_in_leaf'] = min_data_in_leaf
            self.params['max_depth'] = max_depth
            self.params['learning_rate'] = learning_rate
            self.params['feature_fraction'] = feature_fraction
            self.params['bagging_fraction'] = bagging_fraction
            self.params['random_state'] = seed
            self.params['lambda_l2'] = lambda_l2
            self.params['nthread'] = nthread
            self.params.update(extra_params)
            self.num_boost_round = n_estimators
            self.verbose_eval = verbose
            self.early_stopping_rounds = early_stopping

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        train = self.__data_generation(x_train, y_train)
        if not x_valid:
            valid_sets = [train]
        else:
            valid = self.__data_generation(x_valid, y_valid)
            valid_sets = [train, valid]
        self.model = lgb.train(self.params, train, valid_sets=valid_sets,
                               num_boost_round=self.num_boost_round,
                               verbose_eval=self.verbose_eval,
                               early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, x_test):
        return self.model.predict(x_test[self.features])

    def model_saving(self, file_path):
        self.model.save_model(file_path)

    def get_features_importance(self):
        features_importance = pd.DataFrame()
        features_importance['features'] = self.features
        features_importance['gain'] = self.model.feature_importance(type='gain')
        features_importance['split'] = self.model.feature_importance(type='split')
        return features_importance

    def __data_generation(self, data, labels):
        if not self.cate_idx:
            return lgb.Dataset(data[self.features], label=labels, categorical_feature=self.features[self.cate_idx])
        else:
            return lgb.Dataset(data[self.features], label=labels)
