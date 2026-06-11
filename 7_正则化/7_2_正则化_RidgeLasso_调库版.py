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
poly = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly.fit_transform(X.reshape(-1, 1))
# 进行线性回归
from sklearn.linear_model import LinearRegression, Ridge, Lasso
model = LinearRegression()
model.fit(X_poly, y)
pred_liner = model.predict(X_poly)
from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y, pred_liner)
r2 = model.score(X_poly, y)
print(f"普通线性回归 → RMSE={rmse:.2f}, R²={r2:.3f}")
# 创建 Ridge 回归模型
model_rige = Ridge(alpha=10)
model_rige.fit(X_poly, y)
pred_Rige= model_rige.predict(X_poly)
rmse_r = root_mean_squared_error(y, pred_Rige)
r2_r = model_rige.score(X_poly, y)
print(f"Ridge回归(alpha=10) → RMSE={rmse_r:.2f}, R²={r2_r:.3f}")

# 看看系数大小（正则化真正的功劳在这里）
print("\n系数对比：")
print(f"普通线性回归: 最大系数={max(abs(model.coef_)):.2f}")
print(f"Ridge:          最大系数={max(abs(model_rige.coef_)):.2f}")

# ====== 可视化对比 ======
plt.figure(figsize=(10, 6))
# 按 X 排序，保证曲线画出来是顺的
idx = np.argsort(X)
plt.scatter(X, y, color='gray', label='原始数据', s=40)
plt.plot(X[idx], pred_liner[idx], 'b-', linewidth=2, label='普通线性回归')
plt.plot(X[idx], pred_Rige[idx], 'r--', linewidth=2, label=f'Ridge (alpha=10)')
plt.plot(X[idx], y_true[idx], 'g:', linewidth=2, label='真实规律(无噪声)', alpha=0.7)
plt.xlabel("X")
plt.ylabel("y")
plt.title("普通线性回归 vs Ridge 回归")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

