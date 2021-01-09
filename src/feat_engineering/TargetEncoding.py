"""
Target encoding for categorical features (so called likelihood encoding and mean encoding)
The main methods in class TargetEncoding are same as sklearn: fit / transform / fit_transform
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.07
"""
import numpy as np
from sklearn.model_selection import KFold


class TargetEncoding(object):
    def __init__(self, num_outer_folds=5, num_inner_folds=5, random_seed=2020):
        self.outer_fold = KFold(n_splits=num_outer_folds, random_state=random_seed, shuffle=True)
        self.inner_fold = KFold(n_splits=num_inner_folds, random_state=random_seed, shuffle=True)
        self.map_dict = {}

    def fit(self, feat_data, label):  # feat_data must be a vector contains categorical features
        # +++ outer K-Fold loop +++
        outer_target_dict = {c: [] for c in np.unique(feat_data)}
        for NFold, (outer_train_idx, outer_test_idx) in enumerate(self.outer_fold.split(feat_data)):
            # outer in-fold & out-of-fold
            outer_if_feat = feat_data[outer_train_idx]
            outer_if_label = label[outer_train_idx]
            outer_oof_feat = feat_data[outer_test_idx]
            outer_if_label_pre = _kfold_target_encoding(outer_if_feat, outer_if_label, self.inner_fold)

            # calculating outer target scores
            target_dict = {cls: 0 for cls in np.unique(outer_oof_feat)}                 # outer_oof_feat中所有的标签
            target_dict.update(_avg_score_for_cls(outer_if_feat, outer_if_label_pre))   # 根据outer_if_label_pre中的数据对其更新
            for k, v in target_dict.items():
                outer_target_dict[k].append(v)
            print('the %d th outer fold has been finished' % NFold)
        # updating target dict to whole feat data
        self.map_dict = {k: np.mean(val) for k, val in outer_target_dict.items()}

    def transform(self, feat_data):
        label_pre = np.array([self.map_dict[c] for c in feat_data])
        return label_pre

    def fit_transform(self, feat_data, label):
        self.fit(feat_data, label)
        label_pre = self.transform(feat_data)
        return label_pre


def _kfold_target_encoding(feat, label, folder):
    """
    对feat中的类别通过多折交叉验证的方式打分
    :param feat: 类别特征, (n, )
    :param label: 得分标签， (n, )
    :param folder: k折交叉验证对象
    :return: 对feat所有样本的打分列表
    """
    # 存放每一折得到的feat->score映射的得分字典
    map_score_lst = {c: [] for c in np.unique(feat)}
    for train_idx, test_idx in folder.split(feat):
        if_feat = feat[train_idx]
        if_label = label[train_idx]
        oof_feat = feat[test_idx]

        # 根据in-fold中的数据得到的feat->score的映射字典
        target_dict = {cls: 0 for cls in np.unique(oof_feat)}
        target_dict.update(_avg_score_for_cls(if_feat, if_label))
        for k, v in target_dict.items():
            map_score_lst[k].append(v)
    map_dict = {k: np.mean(val) for k, val in map_score_lst.items()}
    label_pre = np.array([map_dict[c] for c in feat])
    return label_pre


def _avg_score_for_cls(feat, score):
    """
    计算feat中每个class对应的score的均值
    :param feat: 类别特征, (n, )
    :param score: 得分标签， (n, )
    :return: 从feat到score的映射字典
    """
    score_dict = {}
    for c, v in zip(feat, score):
        if c in score_dict:
            score_dict[c]["val"] += v
            score_dict[c]["cnt"] += 1
        else:
            score_dict.update({c: {"val": v, "cnt": 1}})
    return {key: val["val"] / val["cnt"] for key, val in score_dict.items()}

