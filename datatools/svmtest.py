import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import gc

import sys

import datatools.datastore_util as du
import datatools.model_util as modelUtil
import datatools.visualize as vis
import pandas as pd
import numpy as np
import util.preprocess_util as ut
import time
from datatools.transform_util import cart2sphere
#
train = du.loadPickleDF("small")
train=train[["TARGET",'AMT_CREDIT', 'AMT_ANNUITY']][0:500]
train=train.dropna()
y=train["TARGET"]
feats=['AMT_CREDIT', 'AMT_ANNUITY']
X=train[feats]

X=X.as_matrix()
y=y.as_matrix()
# we create clusters with 1000 and 100 points
# rng = np.random.RandomState(0)
# n_samples_1 = 1000
# n_samples_2 = 100
# X = np.r_[1.5 * rng.randn(n_samples_1, 2),
#           0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
# y = [0] * (n_samples_1) + [1] * (n_samples_2)

# fit the model and get the separating hyperplane
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# fit the model and get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

# plot separating hyperplanes and samples
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.legend()

# plot the decision functions for both classifiers
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# get the separating hyperplane
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

# get the separating hyperplane for weighted classes
Z = wclf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins for weighted classes
b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
           loc="upper right")
plt.show()