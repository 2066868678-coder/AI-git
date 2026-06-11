"""
多项式回归 - 调库版
==================
用 sklearn.preprocessing.PolynomialFeatures 自动造特征
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 数据（跟手写版一样）======
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_true = 10 + 3 * x - 0.3 * x**2
y = y_true + np.random.randn(11) * 2
poly = PolynomialFeatures(degree= 2, include_bias= False)
X_poly = poly.fit_transform(x.reshape(-1,1))
print(X_poly)