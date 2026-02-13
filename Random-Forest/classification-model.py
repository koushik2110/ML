import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve

pd.set_option('display.max_columns', None)
df = pd.read_csv("travel.csv")
print(df.head())

df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')

df['Age'].fillna(df['Age'].median(), inplace=True)
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df['NumberOfTrips'].fillna(df['NumberOfTrips'].median(),   inplace=True)

df['NumberOfChildrenVisiting'].fillna(df['NumberOfChildrenVisiting'].mode()[0], inplace=True)
df['TypeofContact'].fillna(df['TypeofContact'].mode()[0], inplace=True)
df['DurationOfPitch'].fillna(df['DurationOfPitch'].median(),  inplace=True)
df['NumberOfFollowups'].fillna(df['NumberOfFollowups'].mode()[0], inplace=True)
df['PreferredPropertyStar'].fillna(df['PreferredPropertyStar'].mode()[0], inplace=True)

df.drop(columns=[ 'CustomerID'], inplace=True, axis=1)

df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df.drop(columns=['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)

num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
print('Num of Numerical Features :', len(num_features))

cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
print('Num of Categorical Features :', len(cat_features))

discrete_features=[feature for feature in num_features if len(df[feature].unique())<=25]
print('Num of Discrete Features :',len(discrete_features))

continuous_features=[feature for feature in num_features if feature not in discrete_features]
print('Num of Continuous Features :',len(continuous_features))

X = df.drop(['ProdTaken'], axis=1)
y = df['ProdTaken']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# for a in X.columns:
#     print(df[a].value_counts())
num_features = X.select_dtypes(include='number').columns.tolist()
cat_features = X.select_dtypes(exclude='number').columns.tolist()

preprocessor = ColumnTransformer(
    [
        ('categorical', OneHotEncoder(drop="first"), cat_features),
        ('numerical', StandardScaler(), num_features),
    ]
)

print(pd.DataFrame(train_x).isna().sum())
print(pd.DataFrame(test_x).isna().sum())

train_x = preprocessor.fit_transform(train_x)
test_x = preprocessor.transform(test_x)

print(pd.DataFrame(train_x).isna().sum())
print(pd.DataFrame(test_x).isna().sum())


models = {
    "Logisitic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(train_x, train_y)  # Train model

    # Make predictions
    y_train_pred = model.predict(train_x)
    y_test_pred = model.predict(test_x)

    # Training set performance
    model_train_accuracy = accuracy_score(train_y, y_train_pred)  # Calculate Accuracy
    model_train_f1 = f1_score(train_y, y_train_pred, average='weighted')  # Calculate F1-score
    model_train_precision = precision_score(train_y, y_train_pred)  # Calculate Precision
    model_train_recall = recall_score(train_y, y_train_pred)  # Calculate Recall
    model_train_rocauc_score = roc_auc_score(train_y, y_train_pred)

    # Test set performance
    model_test_accuracy = accuracy_score(test_y, y_test_pred)  # Calculate Accuracy
    model_test_f1 = f1_score(test_y, y_test_pred, average='weighted')  # Calculate F1-score
    model_test_precision = precision_score(test_y, y_test_pred)  # Calculate Precision
    model_test_recall = recall_score(test_y, y_test_pred)  # Calculate Recall
    model_test_rocauc_score = roc_auc_score(test_y, y_test_pred)  # Calculate Roc

    print(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))

    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))

    print('----------------------------------')

    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))

    print('=' * 35)
    print('\n')

## Hyperparameter Training
# rf_params = {"max_depth": [5, 8, 15, None, 10],
#              "max_features": [5, 7, 8],
#              "min_samples_split": [2, 8, 15, 20],
#              "n_estimators": [100, 200, 500, 1000]}
#
# randomcv_models = [
#     ("RF", RandomForestClassifier(), rf_params)
#
# ]
#
# model_param = {}
# for name, model, params in randomcv_models:
#     random = RandomizedSearchCV(estimator=model,
#                                    param_distributions=params,
#                                    n_iter=100,
#                                    cv=3,
#                                    verbose=2,
#                                    n_jobs=-1)
#     random.fit(train_x, train_y)
#     model_param[name] = random.best_params_
#
# for model_name in model_param:
#     print(f"---------------- Best Params for {model_name} -------------------")
#     print(model_param[model_name])

models_1 = {
    "Random Forest": RandomForestClassifier(max_features=8, max_depth=None, n_estimators=500, min_samples_split=2, n_jobs=-1),
}
for i in range(len(list(models_1))):
    model = list(models_1.values())[i]
    model.fit(train_x, train_y)  # Train model

    # Make predictions
    y_train_pred = model.predict(train_x)
    y_test_pred = model.predict(test_x)

    # Training set performance
    model_train_accuracy = accuracy_score(train_y, y_train_pred)  # Calculate Accuracy
    model_train_f1 = f1_score(train_y, y_train_pred, average='weighted')  # Calculate F1-score
    model_train_precision = precision_score(train_y, y_train_pred)  # Calculate Precision
    model_train_recall = recall_score(train_y, y_train_pred)  # Calculate Recall
    model_train_rocauc_score = roc_auc_score(train_y, y_train_pred)

    # Test set performance
    model_test_accuracy = accuracy_score(test_y, y_test_pred)  # Calculate Accuracy
    model_test_f1 = f1_score(test_y, y_test_pred, average='weighted')  # Calculate F1-score
    model_test_precision = precision_score(test_y, y_test_pred)  # Calculate Precision
    model_test_recall = recall_score(test_y, y_test_pred)  # Calculate Recall
    model_test_rocauc_score = roc_auc_score(test_y, y_test_pred)  # Calculate Roc

    print(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))

    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))

    print('----------------------------------')

    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))

    print('=' * 35)
    print('\n')
