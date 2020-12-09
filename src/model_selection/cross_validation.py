# Author Yaoyao Bao
# yaoyaobao@mail.ecust.edu.cn

import pandas as pd


class CrossValidation:
    def __init__(self, model_class, kfold, num_split):
        self.kfold = kfold                  # the KFold split class to split the data into several folds
        self.num_split = num_split          # number of folds
        self.model_class = model_class      # the class of the model
        self.trained_models = []            # save the models trained with different folds

    def train_offline(self, train_feat, train_target):
        """
        train the model while it leaves one fold for model evaluation
        :param train_feat: a dataframe of features
        :param train_target: a dataframe of targets
        :return:
        """
        # transform to pandas.DataFrame
        train_feat = pd.DataFrame(train_feat)
        train_target = pd.DataFrame(train_target)

        validation_res = train_target.copy(deep=True)
        for fold_idx, (train_idx, test_idx) in enumerate(self.kfold(self.num_split).split(train_feat, train_target)):
            model = self.model_class()
            print("##################### Fold {} is running ... #####################".format(fold_idx))
            train_x, test_x = train_feat.iloc[train_idx], train_feat.iloc[test_idx]
            train_y, test_y = train_target.iloc[train_idx], train_target.iloc[test_idx]
            train_idx, val_idx = next(self.kfold(6).split(train_x, train_y))
            train_x, val_x = train_x.iloc[train_idx], train_x.index[val_idx]
            train_y, val_y = train_y.iloc[train_idx], train_y.index[val_idx]
            model.fit(train_x, train_y, val_x, val_y)
            # save the model
            self.trained_models.append(model)
            # oof estimation
            validation_res.iloc[test_idx] = model.predict(test_x)
        return validation_res

    def train_online(self, train_feat, train_target):
        """
        train the model without leave a special set for model evaluation
        :param train_feat: a dataframe of train features
        :param train_target: a dataframe of train targets
        :return:
        """
        # transform to pandas.DataFrame
        train_feat = pd.DataFrame(train_feat)
        train_target = pd.DataFrame(train_target)

        for fold_idx, (train_idx, test_idx) in enumerate(self.kfold(self.num_split).split(train_feat, train_target)):
            model = self.model_class()
            print("##################### Fold {} is running ... #####################".format(fold_idx))
            train_x, val_x = train_feat.iloc[train_idx], train_feat.iloc[test_idx]
            train_y, val_y = train_target.iloc[train_idx], train_target.iloc[test_idx]
            model.fit(train_x, train_y, val_x, val_y)
            # save the model
            self.trained_models.append(model)
        return None

    def predict(self, test_feat):
        """
        make prediction on test_feat
        :param test_feat: a dataframe of features
        :return:
        """
        score = 0
        for model in self.trained_models:
            score += model.predict(test_feat)
        return score / self.num_split
