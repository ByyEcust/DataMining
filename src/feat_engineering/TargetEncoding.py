"""
Target encoding for categorical features (so called likelihood encoding and mean encoding)
The main methods in class TargetEncoding are same as sklearn: fit / transform / fit_transform
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.07
"""
import copy

import numpy as np
from sklearn.model_selection import KFold


class TargetEncoding(object):
    def __init__(self, num_outer_folds=5, num_inner_folds=5, random_seed=2020):
        self.outer_fold = KFold(n_splits=num_outer_folds, random_state=random_seed, shuffle=True)
        self.inner_fold = KFold(n_splits=num_inner_folds, random_state=random_seed, shuffle=True)
        self.target_dict = {}

    def fit(self, feat_data, label):  # feat_data must be a vector contains categorical features
        # +++ outer K-Fold loop +++
        outer_target_dict = self.__target_dict_generation(feat_data)
        for NFold, (outer_train_idx, outer_test_idx) in enumerate(self.outer_fold.split(feat_data)):
            # outer in-fold
            outer_if_feat = copy.deepcopy(feat_data[outer_train_idx])
            outer_if_label = copy.deepcopy(label[outer_train_idx])
            # outer out-of-fold
            outer_oof_feat = copy.deepcopy(feat_data[outer_test_idx])
            # +++ inner K-Fold loop +++
            inner_target_dict = self.__target_dict_generation(feat_data)
            for inner_train_idx, inner_test_idx in self.inner_fold.split(outer_if_feat):
                # inner in-fold
                inner_if_feat = copy.deepcopy(outer_if_feat[inner_train_idx])
                inner_if_label = copy.deepcopy(outer_if_label[inner_train_idx])
                # inner out-of-fold
                inner_oof_feat = copy.deepcopy(outer_if_feat[inner_test_idx])
                # calculating inner target scores
                target_dict = self.__target_score_calculation(inner_oof_feat, inner_if_feat,
                                                              inner_if_label, inner_target_dict)
                inner_target_dict.update(target_dict)
            outer_if_label_pre = self.__dict_2_label(inner_target_dict, outer_if_feat)
            # calculating outer target scores
            target_dict = self.__target_score_calculation(outer_oof_feat, outer_if_feat,
                                                          outer_if_label_pre, outer_target_dict)
            outer_target_dict.update(target_dict)
            print('the %d th outer fold has been finished' % NFold)
        # updating target dict to whole feat data
        self.target_dict.update(outer_target_dict)

    def transform(self, feat_data):
        label_pre = self.__dict_2_label(self.target_dict, feat_data)
        return label_pre

    def fit_transform(self, feat_data, label):
        self.fit(feat_data, label)
        label_pre = self.transform(feat_data)
        return label_pre

    @staticmethod
    # generating a dict to save the results of each class in the categorical feature
    def __target_dict_generation(data):
        return {c: [] for c in np.unique(data)}

    @staticmethod
    # calculating mean value of labels to each relative class of categorical feature
    def __target_score_calculation(feat_data_oof, feat_data_if, label_if, target_dict):
        target_class = np.unique(feat_data_oof)
        for target in target_class:
            target_idx = np.argwhere(feat_data_if == target)
            target_dict[target].append(np.mean(label_if[target_idx]) if len(target_idx) else 0)
        return target_dict

    @staticmethod
    # encoding each class in categorical feature by the dict obtained by func: __target_score_calculation
    def __dict_2_label(target_dict, feat_data):
        map_dict = {key: np.mean(value) for key, value in target_dict.items()}
        return np.array([map_dict[x] for x in feat_data])
