"""
手写 Ridge 回归（带 L2 正则化的线性回归）
======================================
目标：在普通线性回归的损失函数上加"罚款"，防止过拟合
"""
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 生成数据：用多项式特征 + 噪声，故意让普通线性回归过拟合
# 实际规律：y = 0.5 + 2x - 3x² + 5x³
np.random.seed(42)
X = np.linspace(-3, 3, 30)
y_true = 0.5 + 2*X - 3*X**2 + 5*X**3
y = y_true + np.random.randn(30) * 15  # 加噪声，更容易过拟合

# 构造多项式特征（最高到 5 次）
X_poly = np.column_stack([X**i for i in range(1, 6)])  # 5个特征：x, x², x³, x⁴, x⁵
# 在左边加一列 1（偏置项）
X_design = np.column_stack([np.ones(30), X_poly])

print("数据准备好了！X_design 形状:", X_design.shape)
print("前3行数据：")
print(X_design[:3])
w = np.zeros(6)
lambda_ = 10
learning_rata = 0.001
n_literations = 5000
for i in range(n_literations):
    y_pred = X_design @ w
    loss = np.mean((y - y_pred)**2)+lambda_ * np.sum(w[1:]**2)