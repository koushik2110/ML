import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge, Lasso, ElasticNet, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
#
# df = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv', header=1)
# df.loc[:122,"Region"]=0
# df.loc[122:,"Region"]=1
# df['Region'] = df['Region'].astype('int8')
# # df.dropna(inplace=True, axis=0)
# df=df.dropna().reset_index(drop=True)
# # print(df[df['day'] == 'day'])
# df=df.drop(122).reset_index(drop=True)
#
# df.columns = df.columns.str.strip()
# # print(df.columns)
# df[['day','month','year', 'Temperature', 'RH', 'Ws']] = df[['day','month','year', 'Temperature', 'RH', 'Ws']].astype(int )
# objects = [feat for feat in df.columns if df[feat].dtypes == 'O']
#
# for feat in objects:
#     if feat!='Classes':
#         df[feat]=df[feat].astype('float')
#
#
# df['Classes'] = df['Classes'].str.strip()
# # print(df['Classes'].value_counts())
#
# df['Classes'] = np.where(df['Classes']=='not fire',1,0)
# print(df.head())
# df.drop(['day','month','year'], axis=1, inplace=True)
#
# df.to_csv("Algerian_forest_fires_dataset.csv", index=False)


df = pd.read_csv('Algerian_forest_fires_dataset.csv')

# sns.heatmap(df.corr(), annot=True)
X = df.drop(columns=['FWI'])
y = df['FWI']
# print(X.tail())

def correlation(dataset, threshold):
    # print(dataset.head())
    corr_matrix = dataset.corr().abs()*100
    col_corr = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

X.drop(correlation(X, 85), axis=1, inplace=True)

# new_x = pd.DataFrame(StandardScaler().fit_transform(X))
# print(new_x.head())

# fig, axes = plt.subplots(1,2 , figsize = (10,6))
# axes[0].boxplot(X)
# axes[1].boxplot(new_x)

# fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# sns.boxplot(pd.DataFrame(X), ax=axes[0])
# axes[0].set_title("X")
#
# sns.boxplot(pd.DataFrame(new_x), ax=axes[1])
# axes[1].set_title("new_x")

# for a in X.columns:
#     sns.histplot(X[a], kde=True)
#     plt.title(a)
#     plt.tight_layout()
#     plt.show()
# print(X.head())

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)

#linear regression
pipeline1 = Pipeline([('scaler', StandardScaler()),('regressor', LinearRegression())])
pipeline1.fit(train_X, train_y)
predictions1 = pipeline1.predict(test_X)
mse1 = mean_squared_error(test_y, predictions1)
r21 = r2_score(test_y, predictions1)
print('MSE(regressor): %.3f' % mse1)
print('R2(regressor): %.3f' % r21)

#ridge
pipeline2 = Pipeline([('scaler', StandardScaler()),('ridge', Ridge())])
pipeline2.fit(train_X, train_y)
predictions2 = pipeline2.predict(test_X)
mse2 = mean_squared_error(test_y, predictions2)
r22 = r2_score(test_y, predictions2)
print('MSE(ridge): %.3f' % mse2)
print('R2(ridge): %.3f' % r22)


#lasso
pipeline3 = Pipeline([('scaler', StandardScaler()),('lassocv', LassoCV(cv=5, max_iter=1000, random_state=42 ))])
# pipeline3 = Pipeline([('scaler', StandardScaler()),('lasso', Lasso())])
pipeline3.fit(train_X, train_y)
predictions3 = pipeline3.predict(test_X)
mse3 = mean_squared_error(test_y, predictions3)
r23 = r2_score(test_y, predictions3)
print('MSE(lasso): %.3f' % mse3)
print('R2(lasso): %.3f' % r23)
print(pipeline3.named_steps['lassocv'].intercept_)
print(pipeline3.named_steps['lassocv'].coef_)
print(pipeline3.named_steps['lassocv'].alphas_)
print(pipeline3.named_steps['lassocv'].alpha_)


#elasticnet
pipeline4 = Pipeline([('scaler', StandardScaler()),('elasticnet', ElasticNet())])
pipeline4.fit(train_X, train_y)
predictions4 = pipeline4.predict(test_X)
mse4 = mean_squared_error(test_y, predictions4)
r24 = r2_score(test_y, predictions4)
print('MSE(elasticnet): %.3f' % mse4)
print('R2(elasticnet): %.3f' % r24)

# fig, ax = plt.subplots(2, 2)
# sns.scatterplot(x=predictions1, y=test_y, ax=ax[0,0])
# ax[0,0].set_title('Linear regression')
#
# sns.scatterplot(x=predictions2, y=test_y, ax=ax[0,1])
# ax[0,1].set_title('Ridge regression')
#
# sns.scatterplot(x=predictions3, y=test_y, ax=ax[1,0])
# ax[1,0].set_title('Lasso regression')
#
# sns.scatterplot(x=predictions3, y=test_y, ax=ax[1,1])
# ax[1,1].set_title('ElasticNet regression')
#
# plt.show()
