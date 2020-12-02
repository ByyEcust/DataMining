# Author Yaoyao Bao
# yaoyaobao@mail.ecust.edu.cn

from src.feat_engineering.feat_engineering_abstract import FeatEngineeringExtract
import pandas as pd


class ClusterExtract(FeatEngineeringExtract):
    """
    Cluster feat extract class
    To find which cluster the sample belongs to, and encode it with one-hot form

    Parameters
    -------

    agent_type: the name of class which has function <fit> and <transform>,
                e.g. sklearn.cluster.KMeans, sklearn.cluster.SpectralClustering
    init_params: the params that are used to initialize the agent

    """
    def __init__(self, agent_type, **init_params):
        self.agent = agent_type(**init_params)

    def fit(self, feat, **fit_params):
        """
        To fit the model
        :param feat: features in the form of np.array or pd.DataFrame
        :param fit_params: those parameters that are required by agent
        :return:
        """
        self.agent.fit(feat, **fit_params)
        return self

    def transform(self, feat):
        """
        transform the feat into cluster number encoding with one-hot form
        :param feat: np.array or pd.DataFrame, features in table forms
        :return: one-hot features that indicating which cluster the sample belongs (without columns names)
        """
        labels = self.agent.predict(feat)
        cluster_feat = pd.get_dummies(labels)
        if type(feat) == pd.DataFrame:
            cluster_feat.index = feat.index
        return cluster_feat


