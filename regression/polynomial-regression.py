import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

df = pd.read_csv('polinomial-regression.csv')
print(df.head())

# plt.scatter(df['size'], df['price'])
# plt.plot(df['size'], df['price'], color='red')
# plt.show()

train_X, test_X, train_y, test_y = train_test_split(df[['size']], df['price'], test_size=0.2, random_state=42)

model = Pipeline([
    # ('scaler', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=3)),
    ('regressor', LinearRegression())
])

model.fit(train_X, train_y)
predictions = model.predict(test_X)
print(predictions)
r2 = r2_score(test_y, predictions)
print(r2)

sorted_idx = test_X['size'].argsort()
X_sorted = test_X.iloc[sorted_idx]
y_sorted = test_y.iloc[sorted_idx]

plt.scatter(test_X['size'], test_y, label='Test data')
plt.plot(
    X_sorted['size'],
    model.predict(X_sorted),
    color='red',
    linewidth=2,
    label='Polynomial fit (degree=3)'
)

plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Polynomial Regression (Test Data)')
plt.legend()
plt.show()
