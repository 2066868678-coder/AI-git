"""
练习题 - 多项式回归
================
数据都给你准备好了，模型代码你自己写
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 题目1：温度 vs 用电量（U形曲线）======
# 太冷开暖气、太热开空调，用电量两头高中间低
print("=" * 40)
print("题目1：温度 vs 用电量")
temp = np.array([-5, 0, 5, 10, 15, 20, 25, 30, 35, 38])
elec = 15 + 0.3 * (temp - 20)**2 + np.random.randn(10) * 3

# 请自己完成以下步骤：
# 模型预测
poly = PolynomialFeatures(degree= 2 ,include_bias=False)
temp_poly = poly.fit_transform(temp.reshape(-1,1))
model = LinearRegression()
model.fit(temp_poly,elec)
pred = model.predict(temp_poly)
rmse = root_mean_squared_error(pred, elec)
R2  = model.score(temp_poly,elec)
print(f"RMSE = {rmse:.2f}, R^2 = {R2:.3f}")

# 4. 画图：散点图(实际数据) + 红色虚线(模型拟合曲线)
plt.figure(figsize=(8, 5))
plt.scatter(temp, elec, color='blue', label='实际数据', s=50)

# 按温度排序，画出来的线才顺畅
idx = np.argsort(temp)
plt.plot(temp[idx], pred[idx], 'r--', linewidth=2, label='多项式拟合曲线')

plt.xlabel("温度 (°C)")
plt.ylabel("用电量")
plt.title("温度 vs 用电量（多项式回归）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ====== 题目2：试试不同 degree 的效果 ======
print("=" * 40)
print("题目2：对比 degree=1, 2, 3 的效果")
x2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y2_true = 5 + 2 * x2 + 0.5 * x2**2 - 0.1 * x2**3
y2 = y2_true + np.random.randn(10) * 3

# 分别用 degree=1, 2, 3 拟合，对比 RMSE
# 提示：degree=1 就是普通线性回归（一条直线）
#       degree=2 是二次曲线
#       degree=3 是三次曲线
# 看看哪个 RMSE 最小？

degrees = [1, 2, 3]
colors = ['g-', 'r--', 'm-.']
labels = ['一次（直线）', '二次（抛物线）', '三次（S形）']

plt.figure(figsize=(8, 5))
plt.scatter(x2, y2, color='blue', label='实际数据', s=50)

for deg, color, label in zip(degrees, colors, labels):
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    x_poly = poly.fit_transform(x2.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_poly, y2)
    y_pred = model.predict(x_poly)

    rmse = root_mean_squared_error(y_pred, y2)
    r2 = model.score(x_poly, y2)

    print(f"degree={deg} → RMSE={rmse:.2f}, R²={r2:.3f}")

    # 画拟合曲线
    idx = np.argsort(x2)
    plt.plot(x2[idx], y_pred[idx], color, linewidth=2, label=label)

plt.xlabel("x")
plt.ylabel("y")
plt.title("不同多项式次数的拟合效果对比")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

