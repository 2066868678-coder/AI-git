"""
三种回归对比：LinearRegression vs Ridge vs Lasso
=============================================
目的：直观看到正则化（Ridge/Lasso）如何防止过拟合

数据：y = 真实规律 + 噪声，用高次多项式拟合
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. 生成数据：让过拟合明显
# =========================================================
np.random.seed(42)
X = np.linspace(-3, 3, 20)                                   # 故意少一点样本（20个）
y_true = 0.5 + 2*X - 3*X**2 + 5*X**3                         # 真实规律
y = y_true + np.random.randn(20) * 20                         # 加较大噪声

# 构造高次多项式特征（degree=15，特征远多于样本，必然过拟合）
poly = PolynomialFeatures(degree=15, include_bias=False)
X_poly = poly.fit_transform(X.reshape(-1, 1))

# 标准化：把特征统一到同一尺度，Lasso 才能好好收敛
scaler = StandardScaler()
X_poly = scaler.fit_transform(X_poly)

print(f"样本数: {len(X)}, 特征数: {X_poly.shape[1]}")
print("特征比样本多 → 普通线性回归必然过拟合！\n")


# =========================================================
# 2. 训练三个模型
# =========================================================

# --- 2a 普通线性回归（无惩罚）---
model_lr = LinearRegression()
model_lr.fit(X_poly, y)
pred_lr = model_lr.predict(X_poly)

# --- 2b Ridge（L2惩罚，压小系数但不归零）---
model_ridge = Ridge(alpha=50)
model_ridge.fit(X_poly, y)
pred_ridge = model_ridge.predict(X_poly)

# --- 2c Lasso（L1惩罚，把无用系数直接变0）---
model_lasso = Lasso(alpha=5, max_iter=100000)
model_lasso.fit(X_poly, y)
pred_lasso = model_lasso.predict(X_poly)


# =========================================================
# 3. 评估结果
# =========================================================
rmse_lr = root_mean_squared_error(y, pred_lr)
rmse_ridge = root_mean_squared_error(y, pred_ridge)
rmse_lasso = root_mean_squared_error(y, pred_lasso)

print("=" * 50)
print("训练集上的表现（RMSE越小越好）")
print("=" * 50)
print(f"普通线性回归    RMSE = {rmse_lr:.2f}   ← 肯定最小，因为它完全扭曲去拟合每个点")
print(f"Ridge (alpha=50) RMSE = {rmse_ridge:.2f}   ← 大一点，但换来更平滑的曲线")
print(f"Lasso (alpha=5)  RMSE = {rmse_lasso:.2f}   ← 大一点，但把无用的特征剔除了")


# =========================================================
# 4. 系数对比（这才是正则化的核心）
# =========================================================
# 看看用了几个非零系数
n_used_lr = np.sum(model_lr.coef_ != 0)
n_used_ridge = np.sum(np.abs(model_ridge.coef_) > 0.01)     # Ridge不会归零，但接近0
n_used_lasso = np.sum(model_lasso.coef_ != 0)

print("\n" + "=" * 50)
print("系数对比（15个特征的系数大小）")
print("=" * 50)
print(f"普通线性回归: 用了 {n_used_lr}/15 个特征，最大系数 = {max(abs(model_lr.coef_)):.2f}")
print(f"Ridge:        用了 {n_used_ridge}/15 个特征，最大系数 = {max(abs(model_ridge.coef_)):.2f}")
print(f"Lasso:        用了 {n_used_lasso}/15 个特征，最大系数 = {max(abs(model_lasso.coef_)):.2f}")

# 把系数打印出来看看
print("\n具体系数：")
for i in range(min(15, len(model_lr.coef_))):
    print(f"  x^{i+1:2d}: 普通={model_lr.coef_[i]:>12.2f}  "
          f"Ridge={model_ridge.coef_[i]:>8.2f}  Lasso={model_lasso.coef_[i]:>8.2f}")


# =========================================================
# 5. 可视化对比
# =========================================================
plt.figure(figsize=(14, 5))

# --- 图1：拟合曲线对比 ---
plt.subplot(1, 2, 1)
idx = np.argsort(X)
plt.scatter(X, y, color='gray', label='原始数据', s=40, zorder=5)
plt.plot(X[idx], y_true[idx], 'k-', linewidth=2, label='真实规律(无噪声)', alpha=0.5)
plt.plot(X[idx], pred_lr[idx], 'b-', linewidth=1.5, label='普通线性回归')
plt.plot(X[idx], pred_ridge[idx], 'r--', linewidth=2, label='Ridge (alpha=50)')
plt.plot(X[idx], pred_lasso[idx], 'g-.', linewidth=2, label='Lasso (alpha=5)')
plt.xlabel("X")
plt.ylabel("y")
plt.title("拟合曲线对比")
plt.legend()
plt.grid(True, alpha=0.3)

# --- 图2：系数柱状图对比 ---
plt.subplot(1, 2, 2)
x_idx = np.arange(min(10, X_poly.shape[1]))
width = 0.25
plt.bar(x_idx - width, model_lr.coef_[:10], width, label='普通线性回归', alpha=0.7)
plt.bar(x_idx, model_ridge.coef_[:10], width, label='Ridge', alpha=0.7)
plt.bar(x_idx + width, model_lasso.coef_[:10], width, label='Lasso', alpha=0.7)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.xlabel("特征 (x^1 ~ x^10)")
plt.ylabel("系数大小")
plt.title("系数对比（前10个特征）")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('7_3_三种回归对比.png', dpi=150, bbox_inches='tight')
print("\n图片已保存为: 7_3_三种回归对比.png")