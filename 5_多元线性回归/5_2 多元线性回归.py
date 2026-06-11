"""
多元线性回归 - 多因子预测
=========================
用多个特征一起预测一个目标值

场景：房价预测（面积 + 卧室数 + 楼层 → 房价）
"""

import numpy as np


# ========== 数据准备 ==========
area = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
bedrooms = np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4])
floor = np.array([1, 2, 1, 3, 2, 5, 3, 8, 6, 10])

true_price = (0.5 * area + 8 * bedrooms + 2 * floor
              + np.random.randn(10) * 5)
X = np.column_stack([area, bedrooms, floor])
print(X)
X_b = np.c_[np.ones((X.shape[0],1)), X]
print(X_b)
y = true_price
theta = np.random.randn(4) * 0.1
print(theta)
lr = 0.0001
n_iter = 1000
m = len(y)
for i in range(n_iter):
    predictions = X_b.dot(theta)
    errors = predictions - y
    gradient = (2 / m) * X_b.T.dot(errors)
    theta -= lr * gradient
print("训练结果:", theta)
predictions = X_b.dot(theta)
rmse = np.sqrt(np.mean((predictions - y) ** 2))
print(f"RMSE = {rmse:.2f} 万元")

"""
========== ⬇️ 对比：sklearn 三行搞定 ==========
"""
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
sk_pred = model.predict(X)
sk_rmse = np.sqrt(np.mean((sk_pred - y) ** 2))
print(f"\nsklearn 结果: 系数={model.coef_}, 截距={model.intercept_:.2f}")
print(f"sklearn RMSE = {sk_rmse:.2f} 万元")



