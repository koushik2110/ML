import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

X, y = make_classification(n_samples=1000, n_classes=2, n_features=2, random_state=15, n_clusters_per_class=2, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# sns.scatterplot(x = pd.DataFrame(X)[0], y = pd.DataFrame(X)[1], hue=y)
# plt.show()
pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', SVC())
                    ])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

param_grid = [
    {
        'classifier__kernel': ['linear'],
        'classifier__C': [0.1, 1, 10, 100]
    },
    {
        'classifier__kernel': ['rbf'],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': [0.01, 0.1, 1, 10]
    },
    {
        'classifier__kernel': ['poly'],
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': [0.01, 0.1, 1],
        'classifier__degree': [2, 3]
    }
]

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# grid search
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

# predictions
y_pred = grid.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))