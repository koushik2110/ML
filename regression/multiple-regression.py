import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

df = pd.read_csv('economic_index.csv')
df.drop(labels=['Unnamed: 0', 'year', 'month'], axis=1, inplace=True)
train_X, test_X, train_y, test_y = train_test_split(df[['interest_rate','unemployment_rate']], df['index_price'],test_size=0.2, random_state=42)

sns.heatmap(df.corr(), annot=True)
sns.pairplot(df[['interest_rate', 'unemployment_rate']])
plt.show()

model = Pipeline(
    steps=[
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]
)

model.fit(train_X, train_y)
predictions = model.predict(test_X)
print(predictions)
mse = mean_squared_error(test_y, predictions)
r2 = r2_score(test_y, predictions)
mae = mean_absolute_error(test_y, predictions)
