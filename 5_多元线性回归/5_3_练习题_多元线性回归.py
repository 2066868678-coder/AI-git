"""
练习题 - 多元线性回归（sklearn版）
=============================
数据都准备好了，你把模型代码自己写出来
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# ====== 题目1：房价预测 ======
print("=" * 40)
print("题目1：房价预测")
area1 = np.array([40, 55, 65, 78, 85, 95, 105, 118, 125, 138])
bedrooms1 = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 5])
age1 = np.array([20, 15, 10, 8, 5, 3, 2, 1, 6, 12])
price1 = (0.4 * area1 + 10 * bedrooms1 - 0.5 * age1
           + np.random.randn(10) * 8)
X1 = np.column_stack([area1, bedrooms1, age1])
y1 = price1

model = LinearRegression()
model.fit(X1, y1)
pred1 = model.predict(X1)
rmse1 = root_mean_squared_error(y1, pred1)
print("系数:" , model.coef_)
print("R方:", model.score(X1, y1))
print("RMSE:" , f'{rmse1:.2f}万元')

# ====== 题目2：换组数据练手 ======
print("=" * 40)
print("题目2：换一组数据试试")
X2 = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
               [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y2 = 3 * X2[:, 0] + 2 * X2[:, 1] + np.random.randn(10) * 2

model = LinearRegression()
model.fit(X2, y2)
pred2 = model.predict(X2)
rmse2 = root_mean_squared_error(y2, pred2)
print("系数:" , model.coef_)
print("R方:", model.score(X2, y2))
print("RMSE:" , f'{rmse2:.2f}万元')

# ====== 可视化：预测值 vs 真实值（题目1）======
print("=" * 40)
print("画图中...")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 3个特征分别画：真实值 vs 预测值
features = ['面积(平米)', '卧室(间)', '房龄(年)']
X_data = [area1, bedrooms1, age1]

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.scatter(X_data[i], y1, color='blue', label='真实值', alpha=0.7)
    plt.scatter(X_data[i], pred1, color='red', marker='x', label='预测值', alpha=0.7)
    plt.xlabel(features[i])
    plt.ylabel('房价(万元)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

