"""
多元线性回归 - sklearn调库版
=========================
手写已学完，现在学怎么用sklearn 3行搞定
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# ====== 数据（和手写版一样）======
area = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
bedrooms = np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4])
floor = np.array([1, 2, 1, 3, 2, 5, 3, 8, 6, 10])
true_price = (0.5 * area + 8 * bedrooms + 2 * floor
              + np.random.randn(10) * 5)
X = np.column_stack([area, bedrooms, floor])
y = true_price
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print('系数:', model.coef_)
rmse = root_mean_squared_error(y, predictions)
print(f'rmse {rmse:.2f}myuan')