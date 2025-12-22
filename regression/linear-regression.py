import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm


df = pd.read_csv('height-weight.csv')
# sns.heatmap(df.corr(), annot=True)
# plt.show()
print(len(df))
train_X, test_X, train_y, test_y = train_test_split(df[['Weight']], df['Height'], test_size=0.2, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipe.fit(train_X, train_y)
predictions = pipe.predict(test_X)
print(predictions)

# plt.scatter(train_X,train_y)
plt.scatter(test_X,test_y)
# plt.plot(train_X,pipe.predict(train_X))
plt.plot(test_X,predictions)
plt.show()

# After fitting: what the model stores
# Inside pipe:
    # scaler.mean_, scaler.scale_
    # model.coef_
    # model.intercept_
# That’s it.

# print(pipe.named_steps["model"].coef_)
# print(pipe.named_steps["model"].intercept_)

mse = mean_squared_error(test_y, predictions)
r2 = r2_score(test_y, predictions)
mae = mean_absolute_error(test_y, predictions)
# print(mse)
# print(r2)
# print(mae)

model = sm.OLS(train_y,train_X).fit()
prediction1=model.predict(test_X)
print(prediction1)