"""
调库版：Ridge 回归 + Lasso 回归
============================
和普通线性回归对比，看正则化怎么防止过拟合

你自己写所有模型代码（sklearn导入、创建模型、训练、预测、评估）
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 数据准备（我准备）======
np.random.seed(42)
X = np.linspace(-3, 3, 30)                    # 特征
y_true = 0.5 + 2*X - 3*X**2 + 5*X**3          # 真实规律
y = y_true + np.random.randn(30) * 15          # 加噪声后的数据
# =======回归模型===========
# 构造特征
poly = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly.fit_transform(X.reshape(-1, 1))
# 进行回归
from sklearn.linear_model import LinearRegression, Ridge, Lasso

