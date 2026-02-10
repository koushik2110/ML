import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, fbeta_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

X, y = make_classification(n_samples=1000, n_classes=3, n_features=10, random_state=15, n_informative=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model = Pipeline([("logistic", LogisticRegression())])
# model.fit(X_train, y_train)
# prediction = model.predict(X_test)
# cm = confusion_matrix(y_test, prediction)
# classification_report = classification_report(y_test, prediction)
# accuracy = accuracy_score(y_test, prediction)
# recall = recall_score(y_test, prediction, average="weighted")
# precision = precision_score(y_test, prediction, average="weighted")
# fbeta_score = fbeta_score(y_test, prediction, beta=1, average="weighted")
#
# print(cm)
# print(classification_report)
# print(accuracy)
# print(recall)
# print(precision)
# print(fbeta_score)


# ------hyperparameter search--------------
params = [
    {
        "model__solver": ["saga"],
        "model__C": [0.001, 0.01, 0.1, 1, 10],
        "model__l1_ratio": [0.0, 0.5, 1.0]
    }
]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=params,
    cv=skf,
    scoring="accuracy",   # or f1, roc_auc
    n_jobs=-1
)

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
