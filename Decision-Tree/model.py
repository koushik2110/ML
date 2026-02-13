import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import plot_tree

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = DecisionTreeClassifier(ccp_alpha=0.01)
# clf.fit(X_train, y_train)

# plt.figure(figsize = (15,15))
# plot_tree(clf, feature_names = iris.feature_names)
# plt.show()

params = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [1,2,3,4,5,6,7],
    "min_samples_split": [2,3,4,5,6,7],
    "min_samples_leaf": [2,3,4,5,6,7],
    # "max_features": [ "sqrt", "log2"],
    "ccp_alpha": [0.01, 0.1, 1.0],
}

grid = GridSearchCV(clf, param_grid=params, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print(grid.best_params_)
y_pred = grid.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
