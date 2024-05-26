import numpy as np
from scipy.stats import norm, uniform, truncnorm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error as mse
from copy import deepcopy
from scipy import optimize
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from scipy.spatial.distance import cdist


class PlattScaler:

    def __init__(self, rbf_gam):
        super(PlattScaler).__init__()
        self.rbf_gam = rbf_gam
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        self.cal_clf = CalibratedClassifierCV(base_estimator=svm.SVC(gamma=rbf_gam), cv=cv, n_jobs=-1)

    def fit(self, X, y):
        self.cal_clf.fit(X, y)

    def predict_probas(self, dset):
        return self.cal_clf.predict_proba(X=dset)

    def predict(self, dset):
        return self.cal_clf.predict(X=dset)


def nmse(y_true, y_pred):
    return ((y_true / (y_true.mean() + 1e-5) - y_pred / (y_pred.mean() + 1e-5)) ** 2).mean()



