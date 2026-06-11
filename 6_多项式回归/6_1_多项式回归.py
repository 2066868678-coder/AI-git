"""
多项式回归 - 用曲线拟合数据
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 场景：施肥量(kg/亩) vs 农作物产量(kg) ======
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_true = 10 + 3 * x - 0.3 * x**2  # 真实的抛物线关系
y = y_true + np.random.randn(11) * 2  # 加点噪声

# 造多项式特征：把 x 和 x² 并排放成两列
#多项式回归
X_poly = np.column_stack([x, x**2])
model = LinearRegression()
model.fit(X_poly, y)
pred = model.predict(X_poly)
# 训练模型可视化
plt.plot(x, pred,color= 'green',label = '预测曲线')
# 模型评估
rmse = root_mean_squared_error(pred, y)
r2 =model.score(X_poly,y)
# 画数据散点图 + 真实曲线
plt.scatter(x, y, label='实际数据')
plt.plot(x, y_true, 'r--', label='真实曲线')
plt.xlabel('施肥量(kg/亩)')
plt.ylabel('产量(kg)')
plt.legend()
plt.title(f'多项式回归  RMSE={rmse:.2f}kg  R2={r2:.3f}')
plt.show()

