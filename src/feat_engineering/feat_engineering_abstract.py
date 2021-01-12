from abc import abstractmethod


class FeatEngineeringExtract(object):

    @abstractmethod
    def fit(self, feat, label=None, **fit_params):
        return self

    @abstractmethod
    def transform(self, feat):
        return None

    def fit_transform(self, feat, label=None, **fit_params):
        if label is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(feat, **fit_params).transform(feat)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(feat, label, **fit_params).transform(feat)
