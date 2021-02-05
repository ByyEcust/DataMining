'''
The test file of lightGBM and catBoost basic model framework
Author: Ruoqiu Zhang (ECUSTwaterman, waterteam), 2021.01.27
'''
import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from src.model_source.lgb_basic_model import LgbBasicModel
from src.model_source.cat_basic_model import CatBasicModel


# data loading
test_target = 'regression'  # regression / binary / multi

if test_target == 'regression':
    house_price = datasets.load_boston()  # regression
    x = pd.DataFrame(house_price.data, columns=house_price.feature_names)
    y = house_price.target
    features = house_price.feature_names
    loss_fn = mean_squared_error
elif test_target == 'binary':
    cancer = datasets.load_breast_cancer()  # binary classification
    x = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target
    features = cancer.feature_names
    loss_fn = roc_auc_score
else:
    iris = datasets.load_iris()  # multi-labels classification
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    features = iris.feature_names
    loss_fn = cohen_kappa_score

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.20)


# LGB model definition
if test_target == 'regression':
    lgb_model = LgbBasicModel(objective='regression', features=features)
elif test_target == 'binary':
    lgb_model = LgbBasicModel(objective='binary', features=features)
else:
    lgb_model = LgbBasicModel(objective='multiclass', features=features, num_class=3)


# LGB model training
lgb_model.fit(x_train=x_train, y_train=y_train,
              x_valid=x_valid, y_valid=y_valid)


# LGB prediction
train_pre_lgb = lgb_model.predict(x_train)
valid_pre_lgb = lgb_model.predict(x_valid)

print('LGB loss to train: %f' % loss_fn(y_train, train_pre_lgb))
print('LGB loss to valid: %f' % loss_fn(y_valid, valid_pre_lgb))



