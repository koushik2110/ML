from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt



diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

print(X.shape)
print(y.shape)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
params = {
    "min_samples_split": [2,3,4,5,6,7],
    "ccp_alpha": [0.01, 0.1, 1.0],
    "max_depth": [5,10,15,20],
    "min_samples_leaf": [2,3,4,5,6,7],
    "min_weight_fraction_leaf": [0.1,0.2,0.3,0.4],
    "max_leaf_nodes": [2,3,4,5,6,7],
    "min_impurity_decrease": [0.1,0.2,0.3,0.4,0.5,0.6]
}
grid = GridSearchCV(DecisionTreeRegressor(), param_grid=params, cv=5, scoring="neg_mean_squared_error")
grid.fit(train_x, train_y)
print(grid.best_params_)
